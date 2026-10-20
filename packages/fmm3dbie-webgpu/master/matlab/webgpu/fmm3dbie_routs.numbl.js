// Stand-in for the fmm3dbie MEX gateway, which this package does not ship.
// Upstream functions that call it fail here with a message naming what the
// GPU backend does support.
register({
  resolve: function (argTypes, nargout) {
    var outs = [];
    for (var i = 0; i < Math.max(1, nargout); i++) outs.push({ kind: "unknown" });
    return {
      outputTypes: outs,
      apply: function (args) {
        var id = args.length > 0 && args[0] && args[0].kind === "char" ? args[0].value : "";
        var routine = id.split("(")[0] || "this routine";
        throw new RuntimeError(
          "fmm3dbie-webgpu: " + routine + " needs the fmm3dbie MEX, which this package does not have. " +
            "On the GPU, helm3d.impedance.{solver,eval} and helm3d.neumann.{solver,eval} are supported; " +
            "use quad_sphere for geometry."
        );
      },
    };
  },
});
