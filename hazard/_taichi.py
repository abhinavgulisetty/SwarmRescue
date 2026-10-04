"""Taichi kernels loaded lazily by :mod:`hazard.solver`.

This module intentionally does not enable postponed annotations: Taichi 1.7
requires concrete objects in kernel annotations rather than their string form.
"""

import taichi as ti


@ti.kernel
def transport_3d(
    source: ti.types.ndarray(dtype=ti.f64, ndim=3),
    active: ti.types.ndarray(dtype=ti.i32, ndim=3),
    vertical_open: ti.types.ndarray(dtype=ti.i32, ndim=3),
    velocity: ti.types.ndarray(dtype=ti.f64, ndim=4),
    target: ti.types.ndarray(dtype=ti.f64, ndim=3),
    diffusivity: ti.f64,
    dt: ti.f64,
    cell: ti.f64,
):
    """One no-flux first-order-upwind/central-diffusion update."""
    for k, j, i in source:
        center = source[k, j, i]
        if active[k, j, i] == 0:
            target[k, j, i] = center
        else:
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, source.shape[0] - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, source.shape[1] - 1)
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, source.shape[2] - 1)

            z0 = (
                source[km, j, i]
                if k > 0 and vertical_open[k - 1, j, i] != 0
                else center
            )
            z1 = (
                source[kp, j, i]
                if k + 1 < source.shape[0] and vertical_open[k, j, i] != 0
                else center
            )
            y0 = center if active[k, jm, i] == 0 else source[k, jm, i]
            y1 = center if active[k, jp, i] == 0 else source[k, jp, i]
            x0 = center if active[k, j, im] == 0 else source[k, j, im]
            x1 = center if active[k, j, ip] == 0 else source[k, j, ip]

            laplacian = (
                x0 + x1 + y0 + y1 + z0 + z1 - 6.0 * center
            ) / (cell * cell)
            vx = velocity[k, j, i, 0]
            vy = velocity[k, j, i, 1]
            vz = velocity[k, j, i, 2]
            flux_x0 = 0.0
            flux_x1 = 0.0
            flux_y0 = 0.0
            flux_y1 = 0.0
            flux_z0 = 0.0
            flux_z1 = 0.0
            if i > 0 and active[k, j, im] != 0:
                face = 0.5 * (velocity[k, j, im, 0] + vx)
                flux_x0 = face * (x0 if face >= 0.0 else center)
            if i + 1 < source.shape[2] and active[k, j, ip] != 0:
                face = 0.5 * (vx + velocity[k, j, ip, 0])
                flux_x1 = face * (center if face >= 0.0 else x1)
            if j > 0 and active[k, jm, i] != 0:
                face = 0.5 * (velocity[k, jm, i, 1] + vy)
                flux_y0 = face * (y0 if face >= 0.0 else center)
            if j + 1 < source.shape[1] and active[k, jp, i] != 0:
                face = 0.5 * (vy + velocity[k, jp, i, 1])
                flux_y1 = face * (center if face >= 0.0 else y1)
            if k > 0 and vertical_open[k - 1, j, i] != 0:
                face = 0.5 * (velocity[km, j, i, 2] + vz)
                flux_z0 = face * (z0 if face >= 0.0 else center)
            if k + 1 < source.shape[0] and vertical_open[k, j, i] != 0:
                face = 0.5 * (vz + velocity[kp, j, i, 2])
                flux_z1 = face * (center if face >= 0.0 else z1)
            divergence = (
                flux_x1 - flux_x0 + flux_y1 - flux_y0 + flux_z1 - flux_z0
            ) / cell
            target[k, j, i] = center + dt * (
                diffusivity * laplacian - divergence
            )


@ti.kernel
def transport_periodic_1d(
    source: ti.types.ndarray(dtype=ti.f64, ndim=1),
    target: ti.types.ndarray(dtype=ti.f64, ndim=1),
    velocity: ti.f64,
    diffusivity: ti.f64,
    dt: ti.f64,
    cell: ti.f64,
):
    """One periodic update used by the analytic verification test."""
    for i in source:
        before = source[(i - 1 + source.shape[0]) % source.shape[0]]
        after = source[(i + 1) % source.shape[0]]
        gradient = (
            (source[i] - before) / cell
            if velocity >= 0.0
            else (after - source[i]) / cell
        )
        laplacian = (before - 2.0 * source[i] + after) / (cell * cell)
        target[i] = source[i] + dt * (
            diffusivity * laplacian - velocity * gradient
        )
