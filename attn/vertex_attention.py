def vertex_attention(k, v, q, mask=None):
    batch_size, num_vertices, vertex_key_size = k.shape
    batch_size, num_vertices, vertex_value_size = v.shape
    
    if len(q.shape) == 3:
        num_queries, num_heads, _ = q.shape
        assert q.shape == (num_queries, num_heads, vertex_key_size)
        q = q.unsqueeze(0).expand(batch_size, num_queries, num_vertices, num_heads, vertex_key_size)

    if len(q.shape) == 4:
        batch_size, num_queries, num_heads, _ = q.shape
        assert q.shape == (batch_size, num_queries, num_heads, vertex_key_size)
        q = q.unsqueeze(2).expand(batch_size, num_queries, num_vertices, num_heads, vertex_key_size)
    
    assert q.shape == (batch_size, num_queries, num_vertices, num_heads, vertex_key_size)

    k = k.unsqueeze(2).expand(batch_size, num_vertices, num_heads, vertex_key_size)
    assert k.shape == (batch_size, num_vertices, num_heads, vertex_key_size)

    k = k.unsqueeze(1).expand(batch_size, num_queries, num_vertices, num_heads, vertex_key_size)
    assert k.shape == (batch_size, num_queries, num_vertices, num_heads, vertex_key_size)
    assert k.shape == q.shape

    unnormalized_w = (q * k).sum(dim=-1)
    assert unnormalized_w.shape == (batch_size, num_queries, num_vertices, num_heads)

    if mask is not None:
        assert mask.shape == (batch_size, num_vertices)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        mask = mask.expand(batch_size, num_queries, num_vertices, num_heads)
        assert mask.shape == (batch_size, num_queries, num_vertices, num_heads)
        unnormalized_w = unnormalized_w.masked_fill(mask == 0, float("-inf"))
        assert unnormalized_w.shape == (batch_size, num_queries, num_vertices, num_heads)

    w = unnormalized_w.softmax(dim=-2)
    assert w.shape == (batch_size, num_queries, num_vertices, num_heads)

    batch_size, num_vertices, vertex_value_size = v.shape
    v = v.unsqueeze(1).unsqueeze(-2)
    assert v.shape == (batch_size, 1, num_vertices, 1, vertex_value_size)
    v = v.expand(batch_size, num_queries, num_vertices, num_heads, vertex_value_size)
    assert v.shape == (batch_size, num_queries, num_vertices, num_heads, vertex_value_size)

    w = w.unsqueeze(-1)
    assert w.shape == (batch_size, num_queries, num_vertices, num_heads, 1)

    v1 = w * v
    assert v1.shape == (batch_size, num_queries, num_vertices, num_heads, vertex_value_size)
    v1 = v1.sum(dim=2)
    assert v1.shape == (batch_size, num_queries, num_heads, vertex_value_size)

    return v1