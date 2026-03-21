import { defineConfig, loadEnv } from "vite";
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var backendUrl = env.VITE_BACKEND_URL || "http://localhost:7001";
    return {
        server: {
            proxy: {
                "/api": {
                    target: backendUrl,
                    changeOrigin: true,
                    timeout: 10000,
                    proxyTimeout: 10000,
                    configure: function (proxy) {
                        proxy.on("error", function (_error, _req, res) {
                            if (!res.headersSent) {
                                res.writeHead(502, { "Content-Type": "application/json" });
                            }
                            res.end(JSON.stringify({ ok: false, error: "Backend unavailable" }));
                        });
                    },
                },
                "/healthz": {
                    target: backendUrl,
                    changeOrigin: true,
                    timeout: 10000,
                    proxyTimeout: 10000,
                }
            }
        },
        build: {
            outDir: "dist",
            emptyOutDir: true
        }
    };
});
