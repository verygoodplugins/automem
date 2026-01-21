// Type declarations for d3-force-3d
declare module 'd3-force-3d' {
  import type { SimulationNodeDatum, SimulationLinkDatum } from 'd3-force'

  export interface Simulation3D<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum> | undefined
  > {
    restart(): this
    stop(): this
    tick(iterations?: number): this
    nodes(): NodeDatum[]
    nodes(nodes: NodeDatum[]): this
    alpha(): number
    alpha(alpha: number): this
    alphaMin(): number
    alphaMin(min: number): this
    alphaDecay(): number
    alphaDecay(decay: number): this
    alphaTarget(): number
    alphaTarget(target: number): this
    velocityDecay(): number
    velocityDecay(decay: number): this
    force(name: string): any
    force(name: string, force: any): this
    find(x: number, y: number, z?: number, radius?: number): NodeDatum | undefined
    randomSource(): () => number
    randomSource(source: () => number): this
    on(typenames: string): any
    on(typenames: string, listener: any): this
    numDimensions(): number
    numDimensions(nDim: number): this
  }

  export function forceSimulation<NodeDatum extends SimulationNodeDatum>(
    nodes?: NodeDatum[],
    numDimensions?: number
  ): Simulation3D<NodeDatum, undefined>

  export function forceLink<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum>
  >(
    links?: LinkDatum[]
  ): {
    (alpha: number): void
    initialize(nodes: NodeDatum[], random: () => number): void
    links(): LinkDatum[]
    links(links: LinkDatum[]): any
    id(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => any
    id(id: (node: NodeDatum, i: number, nodes: NodeDatum[]) => any): any
    iterations(): number
    iterations(iterations: number): any
    strength(): (link: LinkDatum, i: number, links: LinkDatum[]) => number
    strength(strength: number | ((link: LinkDatum, i: number, links: LinkDatum[]) => number)): any
    distance(): (link: LinkDatum, i: number, links: LinkDatum[]) => number
    distance(distance: number | ((link: LinkDatum, i: number, links: LinkDatum[]) => number)): any
  }

  export function forceManyBody<NodeDatum extends SimulationNodeDatum>(): {
    (alpha: number): void
    initialize(nodes: NodeDatum[], random: () => number): void
    strength(): (d: NodeDatum, i: number, data: NodeDatum[]) => number
    strength(strength: number | ((d: NodeDatum, i: number, data: NodeDatum[]) => number)): any
    distanceMin(): number
    distanceMin(distance: number): any
    distanceMax(): number
    distanceMax(distance: number): any
    theta(): number
    theta(theta: number): any
  }

  export function forceCenter<NodeDatum extends SimulationNodeDatum>(
    x?: number,
    y?: number,
    z?: number
  ): {
    (alpha: number): void
    initialize(nodes: NodeDatum[], random: () => number): void
    x(): number
    x(x: number): any
    y(): number
    y(y: number): any
    z(): number
    z(z: number): any
    strength(): number
    strength(strength: number): any
  }

  export function forceCollide<NodeDatum extends SimulationNodeDatum>(): {
    (alpha: number): void
    initialize(nodes: NodeDatum[], random: () => number): void
    iterations(): number
    iterations(iterations: number): any
    strength(): number
    strength(strength: number): any
    radius(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => number
    radius(radius: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): any
  }

  export function forceRadial<NodeDatum extends SimulationNodeDatum>(
    radius?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number),
    x?: number,
    y?: number,
    z?: number
  ): {
    (alpha: number): void
    initialize(nodes: NodeDatum[], random: () => number): void
    strength(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => number
    strength(strength: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): any
    radius(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => number
    radius(radius: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): any
    x(): number
    x(x: number): any
    y(): number
    y(y: number): any
    z(): number
    z(z: number): any
  }
}
