
    def sum_flux(self, flux):
        scaled_net_flux = np.zeros_like(self.d)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                net_flux = 0
                for idx, (di, dj) in enumerate(self.direction_idx):
                    if (i == 0 and idx == 1) \
                    or (j == 0 and idx == 2) \
                    or (i == self.grid_shape[0] - 1 and idx == 3) \
                    or (j == self.grid_shape[1] - 1 and idx == 0):
                        continue

                    ni, nj = i + di, j + dj
                    if idx in [0, 1]:  # -Q01, -Q02
                        net_flux -= flux[i, j, idx]
                        net_flux -= flux[ni, nj, (idx + 2) % 4]
                    else:  # +Q03, +Q04
                        net_flux += flux[i, j, idx]
                        net_flux += flux[ni, nj, (idx + 2) % 4]

                scaled_net_flux[i, j] = net_flux / (self.dx ** 2)

        return scaled_net_flux

    def water_depth_rk4(self, flux_prev):
        max_iterations = 1
        iteration = 0

        while iteration < max_iterations:
            new_d = np.zeros_like(self.d)
            dd_rk4 = np.zeros_like(self.d)
            iteration += 1
            negative_depth = False
            flow_dir_changed = False

            # k1
            d1 = self.d
            bh_d1 = self.compute_bernoulli_head(self.z, d1, self.u, self.v)
            flow_dir_d1 = self.step1_determine_flow_direction(d1, bh_d1, flux_prev)
            flux_d1 = self.step2_update_mass_flux(flow_dir_d1, bh_d1)
            k1 = self.sum_flux(flux_d1)

            #k2
            d2 = self.d + 0.5 * self.dt * k1
            bh_d2 = self.compute_bernoulli_head(self.z, d2, self.u, self.v)
            flow_dir_d2 = self.step1_determine_flow_direction(d2, bh_d2, flux_prev)
            flux_d2 = self.step2_update_mass_flux(flow_dir_d2, bh_d1)
            k2 = self.sum_flux(flux_d2)

            #k3
            d3 = self.d + 0.5 * self.dt * k2
            bh_d3 = self.compute_bernoulli_head(self.z, d3, self.u, self.v)
            flow_dir_d3 = self.step1_determine_flow_direction(d3, bh_d3, flux_prev)
            flux_d3 = self.step2_update_mass_flux(flow_dir_d3, bh_d3)
            k3 = self.sum_flux(flux_d3)

            #k4
            d4 = self.d + self.dt * k3
            bh_d4 = self.compute_bernoulli_head(self.z, d4, self.u, self.v)
            flow_dir_d4 = self.step1_determine_flow_direction(d4, bh_d4, flux_prev)
            flux_d4 = self.step2_update_mass_flux(flow_dir_d4, bh_d4)
            k4 = self.sum_flux(flux_d4)

            dd_rk4 = (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            new_d = self.d + dd_rk4


        return new_d
