class PDE2D_NS:
    def __init__(self, nx, ny, Lx, Ly, viscosity):
        self.nx = nx  # Number of grid points in x
        self.ny = ny  # Number of grid points in y
        self.Lx = Lx  # Domain size in x
        self.Ly = Ly  # Domain size in y
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.viscosity = viscosity

        # Control positions, every 2 nodes set a control point
        self.control_positions = []
        for i in range(0, nx, 2):
            for j in range(0, ny, 2):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)

        # Control influence matrix B, size (nx*ny, M)
        self.B = np.zeros((nx*ny, self.M))
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def laplacian(self, f):
        """Compute the Laplacian of f using central differences."""
        f = f.reshape(self.nx, self.ny)
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
                (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / self.dx ** 2 +
                (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / self.dy ** 2
        )
        # Neumann BCs (zero normal derivative at boundaries)
        # Left and right boundaries
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        # Bottom and top boundaries
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        return lap.flatten()

    def streamfunction_poisson(self, omega):
        """Solve Poisson equation for streamfunction: ∇²ψ = -ω with Dirichlet BCs ψ=0."""
        omega = omega.reshape(self.nx, self.ny)
        psi = np.zeros_like(omega)
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        dx2dy2 = dx2 * dy2
        denom = 2 * (dx2 + dy2)
        for iteration in range(5000):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = (
                    (dy2 * (psi[2:, 1:-1] + psi[0:-2, 1:-1]) +
                     dx2 * (psi[1:-1, 2:] + psi[1:-1, 0:-2]) +
                     dx2dy2 * (-omega[1:-1, 1:-1])) / denom
            )
            # Enforce Dirichlet boundary conditions ψ=0
            psi[0, :] = 0
            psi[-1, :] = 0
            psi[:, 0] = 0
            psi[:, -1] = 0
            max_diff = np.max(np.abs(psi - psi_old))
            if max_diff < 1e-6:
                break
        return psi.flatten()

    def compute_velocity(self, psi):
        """Compute velocities u and v from streamfunction ψ."""
        psi = psi.reshape(self.nx, self.ny)
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        # Central differences for interior points
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2 * self.dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2 * self.dx)
        # Neumann BCs (zero normal derivative at boundaries)
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]
        return u, v

    def simulate(self, omega0, u_sequence, t_span, time_steps):
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

        N = self.nx * self.ny
        omega = np.zeros((time_steps + 1, N))
        omega[0, :] = omega0.flatten()

        for t in range(time_steps):
            omega_t = omega[t, :]
            u_t = u_sequence[t, :]  # Shape (M,)

            omega_t1 = self.step(omega_t, u_t, dt)
            omega[t + 1, :] = omega_t1

        return omega  # Shape (time_steps + 1, N)

    def step(self, omega, u_t, dt):
        # omega is a flattened array of shape (N,)
        # Add control input
        B_u = self.B @ u_t  # Shape (N,)

        # Reshape to 2D for computation
        omega_2d = omega.reshape(self.nx, self.ny)
        B_u_2d = B_u.reshape(self.nx, self.ny)

        # Solve for streamfunction ψ
        psi = self.streamfunction_poisson(omega)

        # Compute velocities
        u, v = self.compute_velocity(psi)

        # Reshape velocities to 2D
        u = u.reshape(self.nx, self.ny)
        v = v.reshape(self.nx, self.ny)

        # Compute Laplacian of ω
        lap_omega = self.laplacian(omega)

        # Reshape laplacian to 2D
        lap_omega_2d = lap_omega.reshape(self.nx, self.ny)

        # Compute advection term
        conv_omega = np.zeros_like(omega_2d)
        # Central differences for advection
        conv_omega[1:-1, 1:-1] = (
                u[1:-1, 1:-1] * (omega_2d[1:-1, 2:] - omega_2d[1:-1, 0:-2]) / (2 * self.dx) +
                v[1:-1, 1:-1] * (omega_2d[2:, 1:-1] - omega_2d[0:-2, 1:-1]) / (2 * self.dy)
        )

        # Time derivative ∂ω/∂t
        domega_dt = -conv_omega + self.viscosity * lap_omega_2d + B_u_2d

        # Update ω
        omega_new = omega_2d + dt * domega_dt

        return omega_new.flatten()