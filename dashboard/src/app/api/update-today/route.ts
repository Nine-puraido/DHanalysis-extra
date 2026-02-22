import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  const backendDir = path.resolve(process.cwd(), "..", "backend");
  const pythonBin = path.join(backendDir, ".venv", "bin", "python");

  const body = await request.json().catch(() => ({}));
  const days = Math.min(Math.max(Number(body.days) || 1, 1), 7);
  const today = new Date().toISOString().slice(0, 10);

  // Stream progress back via SSE-style newline-delimited JSON
  const encoder = new TextEncoder();
  let closed = false;
  const stream = new ReadableStream({
    async start(controller) {
      function send(data: Record<string, unknown>) {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(JSON.stringify(data) + "\n"));
        } catch { closed = true; }
      }
      function close() {
        if (closed) return;
        try { controller.close(); } catch { /* already closed */ }
        closed = true;
      }

      try {
        // Step 1: Ingest
        send({ step: "ingestion", status: "running", message: `Fetching fixtures for ${days} day(s)...` });
        const ingestionTimeout = 180_000 + days * 120_000; // 3min base + 2min per day (22 leagues = many events)
        const ingestionResult = await execFileAsync(
          pythonBin,
          [
            "-m", "dhx.ingestion.runner",
            "ingest",
            "--source", "sofascore",
            "--date", today,
            "--days", String(days),
          ],
          { cwd: backendDir, timeout: ingestionTimeout }
        );
        send({ step: "ingestion", status: "done", message: "Fixtures ingested", stdout: ingestionResult.stdout.slice(-500) });

        // Step 2: Predict
        send({ step: "prediction", status: "running", message: "Computing features & generating predictions..." });
        const predResult = await execFileAsync(
          pythonBin,
          ["-m", "dhx.modeling", "predict-upcoming"],
          { cwd: backendDir, timeout: 300_000 }
        );
        send({ step: "prediction", status: "done", message: "Predictions generated", stdout: predResult.stdout.slice(-500) });

        send({ step: "complete", status: "done", success: true });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        const stderr =
          err && typeof err === "object" && "stderr" in err
            ? String((err as { stderr: unknown }).stderr).slice(-500)
            : "";
        console.error("update-today failed:", message, stderr);
        send({ step: "error", status: "failed", success: false, error: message, stderr });
      } finally {
        close();
      }
    },
    cancel() { closed = true; },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Transfer-Encoding": "chunked",
      "Cache-Control": "no-cache",
    },
  });
}
