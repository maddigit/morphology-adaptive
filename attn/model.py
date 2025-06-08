import torch
import torch.nn as nn
import json
import os
from .vertex_attention import vertex_attention

class Model(nn.Module):
    def __init__(self, vertex_key_size=2, vertex_value_size=4, muscle_key_size=2, num_heads=20):
        super().__init__()

        self.vertex_key_size = vertex_key_size
        self.vertex_value_size = vertex_value_size
        self.muscle_key_size = muscle_key_size
        self.num_heads = num_heads

        self.muscle_k_to_vertex_q = nn.Sequential(
            nn.Linear(muscle_key_size, 32),
            nn.ReLU(),
            nn.Linear(32, vertex_key_size * num_heads),
        )

        self.wv_to_output = nn.Sequential(
            nn.Linear(num_heads * vertex_value_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, vertex_k, muscle_k, vertex_v, vertex_mask=None):
        batch_size, num_muscles, muscle_key_size = muscle_k.shape
        batch_size, num_vertices, vertex_key_size = vertex_k.shape
        batch_size, num_vertices, vertex_value_size = vertex_v.shape

        q = self.muscle_k_to_vertex_q(muscle_k)
        assert q.shape == (batch_size, num_muscles, self.num_heads * vertex_key_size)
        q = q.view(batch_size, num_muscles, self.num_heads, vertex_key_size)
        assert q.shape == (batch_size, num_muscles, self.num_heads, vertex_key_size)

        wv = vertex_attention(vertex_k, vertex_v, q, mask=vertex_mask)
        assert wv.shape == (batch_size, num_muscles, self.num_heads, vertex_value_size)

        wv = wv.view(batch_size, num_muscles, self.num_heads * vertex_value_size)
        output = self.wv_to_output(wv)
        assert output.shape == (batch_size, num_muscles, 1)
        return output.squeeze(-1)
    
    @staticmethod
    def load(dirname, device="cpu"):
        metadata_filename = os.path.join(dirname, "args.json")

        with open(metadata_filename) as f:
            metadata = json.load(f)
            vertex_key_size = metadata.get("vertex_key_size")
            vertex_value_size = metadata.get("vertex_value_size")
            muscle_key_size = metadata.get("muscle_key_size")
            num_heads = metadata.get("num_heads")

        model = Model(
            vertex_key_size=vertex_key_size,
            vertex_value_size=vertex_value_size,
            muscle_key_size=muscle_key_size,
            num_heads=num_heads
        )

        model_filename = os.path.join(dirname, "model.pt")
        if os.path.exists(model_filename):
            model.load_state_dict(torch.load(model_filename, map_location=device))

        return model

    def save(self, dirname):
        metadata = {
            "vertex_key_size": self.vertex_key_size,
            "vertex_value_size": self.vertex_value_size,
            "muscle_key_size": self.muscle_key_size,
            "num_heads": self.num_heads
        }

        metadata_filename = os.path.join(dirname, "args.json")
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        model_filename = os.path.join(dirname, "model.pt")
        torch.save(self.state_dict(), model_filename)