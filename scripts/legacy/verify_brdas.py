import os
import sys
import random

sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# BRDAS Implementation (copied from api.py for isolated unit testing)
# =====================================================================
class AncestryTuple(tuple):
    def __new__(cls, mu, var, ancestry):
        obj = super().__new__(cls, (mu, var))
        obj.ancestry = ancestry
        return obj


class BrdasList(list):
    def __init__(self, items):
        super().__init__(items)
        self.selections = []
        self.non_bg_classes = [
            'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix',
            'head***ear***lobule', 'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris',
            'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
            'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
            'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
            'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
            'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
            'head***philtrum', 'head***temple', 'head***wrinkles'
        ]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if hasattr(item, 'ancestry'):
            call_idx = len(self.selections)
            region_name = self.non_bg_classes[call_idx] if call_idx < len(self.non_bg_classes) else f"Region {call_idx+1:02d}"
            self.selections.append((region_name, item.ancestry))
        return item


def brdas_sampler(father_pool, mother_pool, father_weight=0.5, mother_weight=0.5):
    num_regions = 33
    sampled_items = []
    
    total_w = father_weight + mother_weight
    father_p = father_weight / total_w if total_w > 0 else 0.5
    
    for _ in range(num_regions):
        if not father_pool and not mother_pool:
            break
        elif not father_pool:
            selected_pool = mother_pool
            ancestry = "Mother"
        elif not mother_pool:
            selected_pool = father_pool
            ancestry = "Father"
        else:
            if random.random() < father_p:
                selected_pool = father_pool
                ancestry = "Father"
            else:
                selected_pool = mother_pool
                ancestry = "Mother"
                
        if selected_pool:
            mu, var = random.choice(selected_pool)
            sampled_items.append(AncestryTuple(mu, var, ancestry))
            
    return BrdasList(sampled_items)

# =====================================================================
# TEST 1: Imbalance Selection Probability (10,000 runs)
# =====================================================================
def test_imbalance_probability():
    print("\n--- Test 1: Dataset Imbalance Independence (10,000 runs) ---")
    father_pool = [("f_mu", "f_var")] * 400
    mother_pool = [("m_mu", "m_var")] * 3
    
    father_selections = 0
    mother_selections = 0
    num_runs = 10000
    
    results_report = []
    
    for run in range(num_runs):
        random_fakes = brdas_sampler(father_pool, mother_pool, father_weight=0.5, mother_weight=0.5)
        
        # Simulate fuse_latent()'s 33 non-background region choices
        for _ in range(33):
            item = random.choice(random_fakes)
            
        f_count = sum(1 for _, a in random_fakes.selections if a == "Father")
        m_count = sum(1 for _, a in random_fakes.selections if a == "Mother")
        
        father_selections += f_count
        mother_selections += m_count
        
        if run < 4:
            results_report.append((run + 1, f_count, m_count))
            
    total_selections = father_selections + mother_selections
    father_pct = (father_selections / total_selections) * 100
    mother_pct = (mother_selections / total_selections) * 100
    
    print("Example Selection Counts per Seed:")
    print("  Test\tFather\tMother")
    for rid, f, m in results_report:
        print(f"  Seed {rid}\t{f}\t{m}")
        
    print(f"\nAverage Selections over {num_runs} runs:")
    print(f"  Father: {father_selections} selections ({father_pct:.2f}%)")
    print(f"  Mother: {mother_selections} selections ({mother_pct:.2f}%)")
    
    assert 49.0 <= father_pct <= 51.0, f"Demographic bias detected: Father got {father_pct:.2f}%"
    assert 49.0 <= mother_pct <= 51.0, f"Demographic bias detected: Mother got {mother_pct:.2f}%"
    print("✅ TEST 1 PASSED: Selection is independent of pool sizes (400 vs 3) and remains exactly 50/50!")

# =====================================================================
# TEST 2: Configurable Weights
# =====================================================================
def test_configurable_weights():
    print("\n--- Test 2: Configurable Weights (70/30) (10,000 runs) ---")
    father_pool = [("f_mu", "f_var")] * 100
    mother_pool = [("m_mu", "m_var")] * 100
    
    father_selections = 0
    mother_selections = 0
    num_runs = 10000
    
    for _ in range(num_runs):
        random_fakes = brdas_sampler(father_pool, mother_pool, father_weight=0.7, mother_weight=0.3)
        for _ in range(33):
            random.choice(random_fakes)
            
        father_selections += sum(1 for _, a in random_fakes.selections if a == "Father")
        mother_selections += sum(1 for _, a in random_fakes.selections if a == "Mother")
        
    total_selections = father_selections + mother_selections
    father_pct = (father_selections / total_selections) * 100
    mother_pct = (mother_selections / total_selections) * 100
    
    print(f"Average Selections (70/30 Config):")
    print(f"  Father: {father_selections} ({father_pct:.2f}%)")
    print(f"  Mother: {mother_selections} ({mother_pct:.2f}%)")
    
    assert 69.0 <= father_pct <= 71.0, f"Weight config failed: Father got {father_pct:.2f}%"
    print("✅ TEST 2 PASSED: Configurable weights behave correctly!")

# =====================================================================
# TEST 3: Deterministic Seed Behaviour (Frozen DNA)
# =====================================================================
def test_deterministic_seed():
    print("\n--- Test 3: Deterministic Seed Behaviour ---")
    father_pool = [("f_mu", "f_var")] * 50
    mother_pool = [("m_mu", "m_var")] * 50
    
    seed = 42
    
    # First run
    random.seed(seed)
    random_fakes_1 = brdas_sampler(father_pool, mother_pool, father_weight=0.5, mother_weight=0.5)
    for _ in range(33):
        random.choice(random_fakes_1)
        
    # Second run
    random.seed(seed)
    random_fakes_2 = brdas_sampler(father_pool, mother_pool, father_weight=0.5, mother_weight=0.5)
    for _ in range(33):
        random.choice(random_fakes_2)
        
    assert random_fakes_1.selections == random_fakes_2.selections, "Deterministic selections failed!"
    print("✅ TEST 3 PASSED: Selections are fully deterministic and frozen under the same seed!")

# =====================================================================
# TEST 4: Same-Race Compatibility (Bypass BRDAS)
# =====================================================================
def mock_gene_factor(encoder, w2sub34, age, gender, race):
    if race == "Indian" and age == "20-29":
        return [("ind_mu", "ind_var")] * 10
    elif race == "EmptyAgeRace" and age == "20-29":
        return []
    elif race == "EmptyAgeRace":
        return [("gen_mu", "gen_var")] * 5
    elif race == "EmptyRace":
        return []
    else:
        return [("gen_mu", "gen_var")] * 5

def query_parent_pools_test(pool_age, gender, race_f, race_m, gene_factor_fn):
    if race_f == race_m:
        entries = gene_factor_fn(None, None, pool_age, gender, race_f)
        if not entries:
            for age in ['0-2', '3-9', '10-19', '20-29']:
                if age != pool_age:
                    entries += gene_factor_fn(None, None, age, gender, race_f)
        return entries

    father_pool = gene_factor_fn(None, None, pool_age, gender, race_f)
    mother_pool = gene_factor_fn(None, None, pool_age, gender, race_m)

    if not father_pool:
        expanded = []
        for age in ['0-2', '3-9', '10-19', '20-29']:
            expanded += gene_factor_fn(None, None, age, gender, race_f)
        father_pool = expanded

    if not mother_pool:
        expanded = []
        for age in ['0-2', '3-9', '10-19', '20-29']:
            expanded += gene_factor_fn(None, None, age, gender, race_m)
        mother_pool = expanded

    return {
        "father_pool": father_pool,
        "mother_pool": mother_pool
    }

def test_same_race_compatibility():
    print("\n--- Test 4: Same-Race Compatibility ---")
    pools = query_parent_pools_test("20-29", "female", "Indian", "Indian", mock_gene_factor)
    
    assert isinstance(pools, list), "Same-race did not bypass BRDAS dict wrapping!"
    assert len(pools) == 10, f"Pool size mismatch: {len(pools)}"
    assert not isinstance(pools, BrdasList), "Same-race returned BrdasList!"
    print("✅ TEST 4 PASSED: Same-race query bypasses BRDAS and returns standard list!")

# =====================================================================
# TEST 5: Age-Expansion Fallback
# =====================================================================
def test_age_expansion():
    print("\n--- Test 5: Age-Expansion Fallback ---")
    pools = query_parent_pools_test("20-29", "female", "EmptyAgeRace", "White", mock_gene_factor)
    
    assert isinstance(pools, dict), "Mixed-race did not return a dictionary!"
    father_pool = pools["father_pool"]
    
    print(f"Expanded Father Pool Size: {len(father_pool)}")
    assert len(father_pool) == 15, f"Age-expansion did not aggregate correctly: expected 15, got {len(father_pool)}"
    print("✅ TEST 5 PASSED: Age-expansion correctly aggregates entries across other age buckets!")


if __name__ == "__main__":
    test_imbalance_probability()
    test_configurable_weights()
    test_deterministic_seed()
    test_same_race_compatibility()
    test_age_expansion()
    print("\n🎉 ALL BRDAS VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉\n")
