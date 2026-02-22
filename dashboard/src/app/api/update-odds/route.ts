import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";

const execFileAsync = promisify(execFile);

export async function POST(request: Request) {
  const backendDir = path.resolve(process.cwd(), "..", "backend");
  const pythonBin = path.join(backendDir, ".venv", "bin", "python");

  const body = await request.json().catch(() => ({}));
  const date =
    typeof body.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(body.date)
      ? body.date
      : new Date().toISOString().slice(0, 10);

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      function send(data: Record<string, unknown>) {
        controller.enqueue(encoder.encode(JSON.stringify(data) + "\n"));
      }

      try {
        send({
          step: "ingestion",
          status: "running",
          message: `Refreshing odds for ${date}...`,
        });
        const result = await execFileAsync(
          pythonBin,
          [
            "-m",
            "dhx.ingestion.runner",
            "ingest",
            "--source",
            "sofascore",
            "--date",
            date,
          ],
          { cwd: backendDir, timeout: 120_000 },
        );
        send({
          step: "ingestion",
          status: "done",
          message: "Odds refreshed",
          stdout: result.stdout.slice(-500),
        });

        send({ step: "complete", status: "done", success: true });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Unknown error";
        const stderr =
          err && typeof err === "object" && "stderr" in err
            ? String((err as { stderr: unknown }).stderr).slice(-500)
            : "";
        console.error("update-odds failed:", message, stderr);
        send({
          step: "error",
          status: "failed",
          success: false,
          error: message,
          stderr,
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Transfer-Encoding": "chunked",
      "Cache-Control": "no-cache",
    },
  });
}
