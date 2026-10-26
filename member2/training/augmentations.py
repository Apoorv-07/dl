import torch


def create_two_views(
    sequence, dropout_probability=0.5, feature_mask_probability=0.15
):
  """Create two stochastic views of the same temporal graph sequence by applying

  edge dropout and feature channel masking.
  """
  view1 = []
  view2 = []

  for graph in sequence:
    # Clone the graph so original is not modified
    graph1 = graph.clone()
    graph2 = graph.clone()

    for g in (graph1, graph2):
      # 1. Apply stochastic Feature Masking to node features (x)
      if hasattr(g, "x") and g.x is not None:
        mask_x = (
            torch.rand_like(g.x, device=g.x.device) > feature_mask_probability
        ).float()
        g.x = g.x * mask_x

      # 2. Apply stochastic Feature Masking to 53 NetFlow edge features (edge_attr)
      if hasattr(g, "edge_attr") and g.edge_attr is not None:
        num_features = g.edge_attr.size(1)
        feat_mask = (
            torch.rand(1, num_features, device=g.edge_attr.device)
            > feature_mask_probability
        ).float()
        g.edge_attr = g.edge_attr * feat_mask

      # 3. Apply Structural Edge Dropout to topology (edge_index) and edge_attr
      if hasattr(g, "edge_index") and g.edge_index is not None:
        num_edges = g.edge_index.size(1)
        keep_mask = (
            torch.rand(num_edges, device=g.edge_index.device)
            > dropout_probability
        )

        g.edge_index = g.edge_index[:, keep_mask]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
          g.edge_attr = g.edge_attr[keep_mask]

    view1.append(graph1)
    view2.append(graph2)

  return view1, view2