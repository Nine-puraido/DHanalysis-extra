import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function createClient() {
  const cookieStore = await cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      db: { schema: "dhx" },
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
      },
      global: {
        fetch: (input, init) =>
          fetch(input, { ...init, cache: "no-store" }),
      },
    }
  );
}
