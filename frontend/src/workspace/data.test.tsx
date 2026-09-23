import { describe, it, expect } from "vitest";
import {
  createDraft,
  csvCell,
  generateNetwork,
  parseCsv,
  requestBody,
  validateDraft,
} from "./data";

describe("workspace input boundaries", () => {
  it("keeps edited demand, windows, fleet capacity and closures in the submitted snapshot", () => {
    const draft = createDraft();
    draft.graph.requests[0] = {
      ...draft.graph.requests[0],
      demand: 73,
      earliest_s: 120,
      latest_s: 300,
      service_s: 25,
    };
    draft.graph.fleet[0].capacity = 81;
    draft.closures = [draft.graph.edges[0].id];
    draft.traffic = 0.5;
    expect(validateDraft(draft)).toBeNull();
    const body = requestBody(draft);
    expect(body.graph.requests[0]).toMatchObject({
      demand: 73,
      earliest_s: 120,
      latest_s: 300,
      service_s: 25,
    });
    expect(body.graph.fleet[0].capacity).toBe(81);
    expect(body.closed_edge_ids).toEqual(draft.closures);
    expect(body.traffic_factor).toBe(0.5);
    draft.graph.requests[0].latest_s = 100;
    expect(validateDraft(draft)).toContain("window end");
  });
  it("parses quoted CSV and rejects incomplete rows", () => {
    expect(
      parseCsv('\uFEFFid,node,note\r\nD1,N1,"Fragile, handle with care"\r\n'),
    ).toEqual([{ id: "D1", node: "N1", note: "Fragile, handle with care" }]);
    expect(() => parseCsv("id,node\nD1")).toThrow("wrong number");
    expect(() => parseCsv('id,node\n"D1,N1')).toThrow("unclosed");
    expect(csvCell("=SUM(A1)")).toBe('"\'=SUM(A1)"');
  });
  it("generates reproducible directed networks with valid referenced nodes", () => {
    const graph = generateNetwork(24, 12, 4, 7);
    expect(graph).toEqual(generateNetwork(24, 12, 4, 7));
    const nodes = new Set(graph.nodes.map((n) => n.id));
    expect(
      graph.edges.every(
        (e) => nodes.has(e.from) && nodes.has(e.to) && e.from !== e.to,
      ),
    ).toBe(true);
    expect(graph.requests.every((r) => nodes.has(r.node))).toBe(true);
  });
});
