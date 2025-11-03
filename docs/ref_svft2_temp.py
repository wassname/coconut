
class TRMSvftLayer(BaseTunerLayer):
    """
    TRM SVFT layer with SVDSteering-style decomposition.
    
    W = U @ S @ V^T + W_res where:
    - U, V: Top-r singular vectors (can be rotated)
    - S: Top-r singular values (can be scaled via dS)
    - W_res: Residual matrix (frozen)
    """

    adapter_layer_names = ("svft_delta_s", "svft_loglambda_s", "svft_rotation_params_u", "svft_rotation_params_v")
    other_param_names = ("svft_u", "svft_v", "svft_s", "svft_w_res", "svft_scale_s", "svft_alpha", "svft_r", "svft_rotate_u", "svft_rotate_v", "svft_rotation_method", "svft_block_size")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.svft_r = {}
        self.svft_rotate_u = {}
        self.svft_rotate_v = {}
        self.svft_rotation_method = {}
        self.svft_block_size = {}
        self.svft_scale_s = {}
        self.svft_alpha = {}
        # self.svft_steer_s = {}
        
        # SVD components (per adapter) - simplified naming like SVDSteering
        self.svft_u = BufferDict({})  # U: [d_out, r]
        self.svft_v = BufferDict({})  # V: [d_in, r]
        self.svft_s = BufferDict({})  # S: [r]
        self.svft_w_res = BufferDict({})  # W_res: [d_out, d_in]
        
        # Learnable S scaling (DeLoRA-style)
        self.svft_delta_s = nn.ParameterDict({})  # add: S + delta_s
        self.svft_loglambda_s = nn.ParameterDict({})  # mult: lambda_s * S
        
        # Rotation parameters (SVDSteering-style)
        self.svft_rotation_params_u = nn.ParameterDict({})
        self.svft_rotation_params_v = nn.ParameterDict({})

        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find TRM layers
        self._recursion_cache = None

        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        scale_s,
        alpha,
        r,
        rotate_u,
        rotate_v,
        rotation_method,
        block_size,
        # steer_s,
        **kwargs
    ) -> None:
        """
        Initialize SVFT adapter with simple top-r SVD + residual (PiSSA-style).
        """
        if adapter_name in self.svft_u:
            return  # Already initialized

        self.svft_scale_s[adapter_name] = scale_s
        self.svft_alpha[adapter_name] = float(alpha)
        self.svft_r[adapter_name] = r
        self.svft_rotate_u[adapter_name] = rotate_u
        self.svft_rotate_v[adapter_name] = rotate_v
        self.svft_rotation_method[adapter_name] = rotation_method
        self.svft_block_size[adapter_name] = block_size
        # self.svft_steer_s[adapter_name] = steer_s

        # Get base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device

        # Simple top-r SVD (like SVDSteering snippet)
        U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
        
        U = U_full[:, :r]  # [d_out, r]
        S = S_full[:r]     # [r]
        Vh = Vh_full[:r, :]  # [r, d_in]
        V = Vh.T           # [d_in, r]
        
        # Compute residual (PiSSA-style)
        W_principal = U @ torch.diag(S) @ Vh
        W_res = base_weight - W_principal
        
        # Store frozen components
        self.svft_u[adapter_name] = U.clone().detach().contiguous()
        self.svft_v[adapter_name] = V.clone().detach().contiguous()
        self.svft_s[adapter_name] = S.clone().detach().contiguous()
        self.svft_w_res[adapter_name] = W_res.clone().detach().contiguous()
        
        # Learnable S scaling (DeLoRA-style)
        if scale_s == "add":
            self.svft_delta_s[adapter_name] = nn.Parameter(
                torch.zeros(r, device=device), 
                requires_grad=True
            )
            nn.init.uniform_(self.svft_delta_s[adapter_name], a=1e-5, b=1e-3)
        elif scale_s == "mult":
            self.svft_loglambda_s[adapter_name] = nn.Parameter(
                torch.zeros(r, device=device), 
                requires_grad=True
            )
        
        # Initialize rotation parameters (SVDSteering-style)
        if rotate_u:
            if rotation_method == "block_diagonal":
                assert block_size is not None and r % block_size == 0, f"block_size {block_size} must divide r {r}"
                num_blocks = r // block_size
                self.svft_rotation_params_u[adapter_name] = nn.Parameter(
                    torch.zeros(num_blocks, block_size, block_size, device=device)
                )
            else:
                self.svft_rotation_params_u[adapter_name] = nn.Parameter(
                    torch.zeros(r, r, device=device)
                )
        
        if rotate_v:
            if rotation_method == "block_diagonal":
                assert block_size is not None and r % block_size == 0, f"block_size {block_size} must divide r {r}"
                num_blocks = r // block_size
                self.svft_rotation_params_v[adapter_name] = nn.Parameter(
                    torch.zeros(num_blocks, block_size, block_size, device=device)
                )
            else:
                self.svft_rotation_params_v[adapter_name] = nn.Parameter(
                    torch.zeros(r, r, device=device)
                )
    def _get_rotation(
        self, 
        params: Float[Tensor, "... r r"],
        alpha: float,
        rotation_method: str,
    ) -> Float[Tensor, "... r r"]:
        """Compute rotation matrix from learnable parameters (SVDSteering-style).
        
        Args:
            params: Rotation parameters (unconstrained)
            alpha: Steering coefficient (1.0 = forward, -1.0 = reverse)
            rotation_method: Rotation parameterization method
        
        Returns:
            Orthogonal rotation matrix R ∈ SO(r)
        """
        if rotation_method == "block_diagonal":
            # params shape: [num_blocks, block_size, block_size]
            blocks = []
            for block_params in params:
                A = block_params - block_params.T  # skew-symmetric
                R_block = self._rotation_from_skew(A, alpha, rotation_method)
                blocks.append(R_block)
            return torch.block_diag(*blocks)
        else:
            # Full rotation: params shape: [r, r]
            A = params - params.T  # skew-symmetric projection
            return self._rotation_from_skew(A, alpha, rotation_method)
    
    def _rotation_from_skew(
        self,
        A: Float[Tensor, "r r"],
        alpha: float,
        rotation_method: str,
    ) -> Float[Tensor, "r r"]:
        """Compute rotation from skew-symmetric matrix."""
        if rotation_method in ["matrix_exp", "block_diagonal"]:
            # Matrix exponential: exp(αA)
            return torch.matrix_exp(alpha * A)
        elif rotation_method == "cayley":
            # Cayley transform: (I - αA/2)^{-1} (I + αA/2)
            # More efficient than matrix_exp, same reversibility
            I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            X = alpha * A / 2
            return torch.linalg.solve(I - X, I + X)
        else:
            raise ValueError(f"Unknown rotation method: {rotation_method}")

    def get_adapted_output(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter output (SVDSteering-style).
        
        W_adapted = U_rot @ diag(S_scaled) @ V_rot^T + W_res
        Forward: x @ V_rot @ diag(S_scaled) @ U_rot^T + x @ W_res^T
        
        Note: alpha scales rotations only (steering strength), not S
        """
        alpha = self.svft_alpha[adapter]
        # steer_s = self.svft_steer_s[adapter]
        
        # Get frozen bases
        U = self.svft_u[adapter]  # [d_out, r]
        V = self.svft_v[adapter]  # [d_in, r]
        S = self.svft_s[adapter]  # [r]
        W_res = self.svft_w_res[adapter]  # [d_out, d_in]
        
        # Apply rotations (alpha scales rotation strength, not magnitude)
        if self.svft_rotate_v[adapter] and adapter in self.svft_rotation_params_v:
            R_v = self._get_rotation(
                self.svft_rotation_params_v[adapter], 
                alpha=alpha,
                rotation_method=self.svft_rotation_method[adapter]
            )
            V_rot = V @ R_v  # [d_in, r]
        else:
            V_rot = V
        
        if self.svft_rotate_u[adapter] and adapter in self.svft_rotation_params_u:
            R_u = self._get_rotation(
                self.svft_rotation_params_u[adapter],
                alpha=alpha,
                rotation_method=self.svft_rotation_method[adapter]
            )
            U_rot = U @ R_u  # [d_out, r]
        else:
            U_rot = U
        
        # Scale S independently (no alpha - this controls magnitude, not direction)
        scale_mode = self.svft_scale_s[adapter]
        if scale_mode == "add":
            delta_s = self.svft_delta_s[adapter]  # [r]
            # if steer_s:
            #     delta_s = delta_s * alpha
            # S_scaled = S + delta_s

            # OR
            S_scaled = S + alpha * torch.tanh(delta_s) * S
        elif scale_mode == "mult":
            loglambda_s = self.svft_loglambda_s[adapter]
            S_scaled = (loglambda_s * alpha).exp() * S
        else:  # "none"
            S_scaled = S
        
        # Efficient forward: x @ V_rot @ diag(S_scaled) @ U_rot^T
        x_projected = x @ V_rot  # [..., r]
        x_scaled = x_projected * S_scaled  # [..., r] - broadcast multiply
        x_transformed = x_scaled @ U_rot.T  # [..., d_out]
        
        # Add residual contribution
        x_residual = x @ W_res.T  # [..., d_out]
        
        return x_transformed + x_residual

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
        assert len(self.active_adapters) <= 1, "TRM SVFT currently supports only one active adapter at a time."

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            # Always compute full adapted weight (no mode switching)
            result = None
            for adapter in self.active_adapters:
                if adapter not in self.svft_u:
                    continue

                h = self.get_adapted_output(x, adapter)
                
                if result is None:
                    result = h
                else:
                    result += h  # Multiple adapters (unlikely)
            
            if result is None:
                result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM SVFT yet")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerge not implemented for TRM SVFT yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmsvft." + rep
