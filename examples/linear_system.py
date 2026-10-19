"""通用线性耦合示例：U_z=V、V_t=U，解析解U=V=exp(z+t)。"""
import json
import numpy as np
from causalwave import march_system


def main():
    z = np.linspace(0, 1, 41)
    t = np.linspace(0, 1, 61)
    shape = (1, 1, len(z), len(t))
    one = np.ones(shape)
    zero = np.zeros(shape)
    source = np.zeros((1, len(z), len(t)))
    U, V = march_system(
        z, t, np.exp(t)[None], np.exp(z)[None],
        one, source, zero, one, source, z_factor=2, t_factor=2,
    )
    exact = np.exp(z[:, None] + t[None])
    errors = [float(np.max(np.abs(field[0] - exact))) for field in (U, V)]
    # 只验收这个明确的解析算例，不将门槛推广为一般精度保证。
    if not np.isfinite(errors).all() or max(errors) >= 1e-3:
        raise RuntimeError(f"解析算例未通过：{errors}")
    print(json.dumps({"equations": "U_z=V, V_t=U", "shape_U": list(U.shape),
                      "shape_V": list(V.shape), "max_abs_U": errors[0],
                      "max_abs_V": errors[1], "passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
