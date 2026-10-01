import { API_BASE_URL } from "./api";
import type { StudentTermFeatures } from "./types";

export function toCurl(features: StudentTermFeatures): string {
  const body = JSON.stringify(features, null, 2);
  return [
    `curl -X POST ${API_BASE_URL}/predict \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '${body}'`,
  ].join("\n");
}
