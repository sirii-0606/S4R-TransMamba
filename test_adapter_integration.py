import torch
from boltz.model.models.boltz2 import Boltz2
from boltz.model.layers.adapter import Adapter
from boltz.model.layers.mamba import MambaBlock

def test_adapter_instantiation():
    print("Testing Adapter instantiation...")
    adapter = Adapter(d_model=128, bottleneck_dim=32)
    x = torch.randn(1, 128)
    out = adapter(x)
    assert out.shape == x.shape
    print("Adapter instantiation passed!")

def test_mamba_block_with_adapter():
    print("Testing MambaBlock with Adapter...")
    block = MambaBlock(d_model=128, use_adapter=True, adapter_dim=32)
    assert hasattr(block, 'adapter')
    assert isinstance(block.adapter, Adapter)
    
    x = torch.randn(2, 10, 128) # (B, L, D)
    out = block(x)
    assert out.shape == x.shape
    print("MambaBlock with Adapter passed!")

def test_boltz2_with_adapter():
    print("Testing Boltz2 with Adapter configuration...")
    # Mocking necessary args for Boltz2
    model = Boltz2(
        atom_s=128,
        atom_z=64,
        token_s=128,
        token_z=64,
        num_bins=32,
        training_args={},
        validation_args={},
        embedder_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 4,
        },
        msa_args={
            "s_dropout": 0.1,
            "z_dropout": 0.1,
            "num_blocks": 1,
            "msa_dropout": 0.1,
            "pair_dropout": 0.1,
            "msa_s": 64,
            "msa_blocks": 1,
        },
        pairformer_args={"num_blocks": 1},
        score_model_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 4,
            "token_transformer_depth": 1,
            "token_transformer_heads": 4,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 4,
            "conditioning_transition_layers": 1,
        },
        diffusion_process_args={},
        diffusion_loss_args={},
        confidence_prediction=False,
        use_mamba_trunk=True,
        use_adapter=True,
        adapter_dim=32,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    )
    
    # Check if adapter is present in the hierarchy
    found_adapter = False
    for name, module in model.named_modules():
        if isinstance(module, Adapter):
            found_adapter = True
            break
            
    assert found_adapter, "Adapter not found in Boltz2 model!"
    print("Boltz2 with Adapter passed!")

if __name__ == "__main__":
    test_adapter_instantiation()
    test_mamba_block_with_adapter()
    test_boltz2_with_adapter()
    print("All tests passed!")
