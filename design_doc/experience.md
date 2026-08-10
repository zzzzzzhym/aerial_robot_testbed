Blade parameter fitting:
- Prefer nested optimization: solve local (v_i), then fit the four global blade parameters.
- Flattening (solve v_i and finding blade params together in one optimization problem) is a bad idea. It creates thousands of variables (each segment of blade has a v_i to solve). This makes least square matrix very large. 