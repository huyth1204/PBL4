"""
Dựng đồ thị đa tầng G_τ (Space + Air + Ground). Không sửa graph_builder.py:
file này tái sử dụng hằng số và công thức trễ d_ij có sẵn ở đó.

Mỗi node có thuộc tính: layer, kind, can_relay, endpoint.
Mỗi cạnh có: delay_ms, bandwidth_mbps, packet_loss, link_type.
"""
from __future__ import annotations

import networkx as nx

from src.graph.graph_builder import (
    BANDWIDTH_GSL_MBPS,
    BANDWIDTH_ISL_MBPS,
    PACKET_LOSS_GSL,
    PACKET_LOSS_ISL,
    total_link_delay_ms,
)
from src.physics.air_ground import Node

# Liên kết Air <-> Vệ tinh (feeder/backhaul). Giá trị GIẢ ĐỊNH.
BANDWIDTH_ASL_MBPS = 100.0
PACKET_LOSS_ASL = 0.01

ALL_INFRA = ("space", "haps", "uav")


def is_kind_enabled(kind: str, infra: set[str] | tuple[str, ...]) -> bool:
    """gateway/user luôn bật (là điểm đầu/cuối). haps/uav/space bật theo `infra`
    -> dùng để chạy ablation: --infra space | space,haps | space,haps,uav."""
    if kind in ("gateway", "user"):
        return True
    return kind in infra


def build_graph_multilayer(
    sat_names: list[str],
    nodes: list[Node],
    gsl_links: list[dict],
    isl_links: list[dict],
    local_links: list[dict],
    infra: set[str] | tuple[str, ...] = ALL_INFRA,
) -> nx.DiGraph:
    G = nx.DiGraph()
    by_name = {n.name: n for n in nodes}

    for s in sat_names:
        G.add_node(s, layer="space", kind="leo", can_relay=True, endpoint=False)
    for n in nodes:
        # Nút bị tắt vẫn có mặt (để N và thứ tự node cố định) nhưng cô lập, không là endpoint.
        G.add_node(
            n.name, layer=n.layer, kind=n.kind, can_relay=n.can_relay,
            endpoint=bool(n.endpoint and is_kind_enabled(n.kind, infra)),
        )

    def add_bidirectional(u, v, dist_km, bw, loss, link_type_up, link_type_down):
        delay = total_link_delay_ms(dist_km, bw)
        G.add_edge(u, v, delay_ms=delay, bandwidth_mbps=bw, packet_loss=loss, link_type=link_type_up)
        G.add_edge(v, u, delay_ms=delay, bandwidth_mbps=bw, packet_loss=loss, link_type=link_type_down)

    # --- Node (ground/air) <-> Vệ tinh: dùng kết quả compute_gsl_links ---
    if "space" in infra:
        for link in gsl_links:
            if not link["exists"]:
                continue
            node = by_name[link["ground_station"]]
            if not is_kind_enabled(node.kind, infra):
                continue
            if node.layer == "ground":
                bw, loss, tag = BANDWIDTH_GSL_MBPS, PACKET_LOSS_GSL, "GSL"
            else:
                bw, loss, tag = BANDWIDTH_ASL_MBPS, PACKET_LOSS_ASL, "ASL"
            add_bidirectional(
                node.name, link["satellite"], link["slant_range_km"], bw, loss,
                f"{tag}_uplink", f"{tag}_downlink",
            )
        for link in isl_links:
            if link["exists"]:
                add_bidirectional(
                    link["sat_a"], link["sat_b"], link["distance_km"],
                    BANDWIDTH_ISL_MBPS, PACKET_LOSS_ISL, "ISL", "ISL",
                )

    # --- Liên kết cục bộ: HAPS-Ground, UAV-Ground, Air-Air ---
    for link in local_links:
        if not link["exists"]:
            continue
        a, b = by_name[link["node_a"]], by_name[link["node_b"]]
        if not (is_kind_enabled(a.kind, infra) and is_kind_enabled(b.kind, infra)):
            continue
        add_bidirectional(
            a.name, b.name, link["distance_km"], link["bandwidth_mbps"],
            link["packet_loss"], link["link_type"], link["link_type"],
        )
    return G


def single_source_relay_dijkstra(G: nx.DiGraph, src: str, weight: str = "delay_ms"):
    """Dijkstra từ `src`, nhưng cạnh đi RA từ node can_relay=False bị ẩn
    (trừ khi đó chính là src). => user terminal chỉ làm điểm đầu/cuối,
    không bao giờ nằm giữa đường đi."""
    can_relay = nx.get_node_attributes(G, "can_relay")

    def w(u, v, data):
        if u != src and not can_relay.get(u, True):
            return None  # networkx: None = cạnh bị ẩn
        return data[weight]

    return nx.single_source_dijkstra(G, src, weight=w)
