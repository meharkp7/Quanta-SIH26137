import { createClient } from "@supabase/supabase-js";
const url = import.meta.env.VITE_SUPABASE_URL?.trim();
const key = (
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ||
  import.meta.env.VITE_SUPABASE_ANON_KEY
)?.trim();
export const supabase =
  url && key
    ? createClient(url, key, {
        auth: {
          persistSession: true,
          autoRefreshToken: true,
          detectSessionInUrl: true,
        },
      })
    : null;
let identity: { token?: string; company?: string; demo?: boolean } = {};
export function setRequestIdentity(value: typeof identity) {
  identity = value;
}
export function identityHeaders(): Record<string, string> {
  return identity.demo
    ? { "X-Quanta-Demo": "true" }
    : {
        ...(identity.token
          ? { Authorization: `Bearer ${identity.token}` }
          : {}),
        ...(identity.company ? { "X-Company-Id": identity.company } : {}),
      };
}
