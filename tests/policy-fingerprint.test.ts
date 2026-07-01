import { describe, expect, it } from "vitest";
import { fingerprintPolicy } from "../src/isolate/policy-fingerprint";
import type { CompiledPolicy } from "../src/egress/types";

const basePolicy = (overrides: Partial<CompiledPolicy> = {}): CompiledPolicy => ({
  policyId: "pol_test",
  policyName: "test",
  allow: ["api.example.com"],
  deny: [],
  headerInjections: [],
  proxy: null,
  vpcRoutes: [],
  ...overrides,
});

describe("fingerprintPolicy", () => {
  it("returns a sha256 digest that does not leak header-injection secrets", async () => {
    const fp = await fingerprintPolicy(
      basePolicy({
        headerInjections: [
          {
            target: "api.example.com",
            header: "authorization",
            secretValue: "sk-live-do-not-leak",
          },
        ],
      }),
    );

    expect(fp).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(fp).not.toContain("sk-live-do-not-leak");
  });

  it("changes when a header-injection secret rotates", async () => {
    const before = await fingerprintPolicy(
      basePolicy({
        headerInjections: [
          { target: "api.example.com", header: "authorization", secretValue: "old" },
        ],
      }),
    );
    const after = await fingerprintPolicy(
      basePolicy({
        headerInjections: [
          { target: "api.example.com", header: "authorization", secretValue: "new" },
        ],
      }),
    );

    expect(after).not.toBe(before);
  });

  it("does not leak proxy code or proxy secrets, and changes when they rotate", async () => {
    const before = await fingerprintPolicy(
      basePolicy({
        proxy: {
          policyId: "pol_test",
          code: "export default { fetch: () => fetch('https://internal.example') }",
          secrets: { AUTHORIZATION: "proxy-secret-old" },
        },
      }),
    );

    expect(before).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(before).not.toContain("internal.example");
    expect(before).not.toContain("proxy-secret-old");

    const after = await fingerprintPolicy(
      basePolicy({
        proxy: {
          policyId: "pol_test",
          code: "export default { fetch: () => fetch('https://internal.example') }",
          secrets: { AUTHORIZATION: "proxy-secret-new" },
        },
      }),
    );

    expect(after).not.toBe(before);
  });

  it("returns the no-policy sentinel for null", async () => {
    expect(await fingerprintPolicy(null)).toBe("(none)");
  });
});
