import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RiskBadge from "./RiskBadge";

describe("RiskBadge", () => {
  it.each([
    ["low", "LOW RISK"],
    ["medium", "MEDIUM RISK"],
    ["high", "HIGH RISK"],
  ] as const)("renders %s as %s", (category, expectedText) => {
    render(<RiskBadge category={category} />);
    expect(screen.getByText(expectedText)).toBeInTheDocument();
  });
});
