import json
import random
import hashlib
import os
from typing import Dict, List, Tuple
from collections import defaultdict

# ----------------------------
# Broad ontology (slots are categories, not narrow items)
# ----------------------------
ONTOLOGY = {
    "Technology": [
        "neural networks", "edge computing", "blockchain", "quantum computing",
        "autonomous systems", "computer vision", "large language models", "smart grids"
    ],
    "Science": [
        "CRISPR gene editing", "photosynthesis", "thermodynamics",
        "radiofrequency communication", "materials science", "epidemiology",
        "Earth observation"
    ],
    "HealthCondition": [
        "cancer", "diabetes", "cardiovascular disease", "depression",
        "anxiety", "respiratory illness"
    ],
    "EnvironmentIssue": [
        "air pollution", "water scarcity", "deforestation", "urban heat islands",
        "coastal flooding", "wildfires"
    ],
    "EconomicConcept": [
        "inflation", "supply and demand", "credit scoring", "risk management",
        "behavioral biases", "labor productivity"
    ],
    "CulturalPractice": [
        "tea ceremonies", "street festivals", "artisan crafts",
        "traditional cuisine", "folk music"
    ],
    "ArtForm": [
        "Impressionist painting", "classical music", "surrealist film",
        "modern sculpture", "photography"
    ],
    "PolicyInstrument": [
        "data protection regulation", "carbon pricing", "public health mandates",
        "algorithmic accountability laws", "antitrust enforcement"
    ],
    "EthicalIssue": [
        "algorithmic bias", "surveillance", "biometric privacy",
        "informed consent", "digital divide"
    ],
    "EducationTheme": [
        "STEM education", "digital literacy", "curriculum design",
        "assessment methods", "inclusive pedagogy"
    ],
    "Sector": [
        "healthcare", "finance", "transportation", "agriculture", "e-commerce",
        "manufacturing", "media and entertainment", "public administration"
    ],
    "Location": {
        "Country": ["Japan", "Italy", "India", "Brazil", "USA", "Germany", "Morocco", "South Korea"],
        "Region": ["Scandinavia", "the Mediterranean", "East Asia", "West Africa", "the Andes"],
        "City": ["Singapore", "Barcelona", "Nairobi", "Vancouver", "Mumbai"],
        "RuralArea": ["a coastal village", "a mountain hamlet", "a desert oasis", "a river valley"]
    },
    "PopulationGroup": [
        "commuters", "patients", "students", "factory workers",
        "small business owners", "remote teams", "teenagers", "older adults"
    ],
    "ActorRole": [
        "Economist", "Ethicist", "Psychologist", "Robotics Engineer",
        "Data Scientist", "Urban Planner", "Public Health Researcher", "Privacy Lawyer"
    ],
    "ImpactDomain": [
        "mental health", "privacy", "employment", "household energy use",
        "education access", "community trust", "environmental quality"
    ],
}

# ----------------------------
# Relation-driven templates (generic; slots specify ontology categories)
# ----------------------------
TEMPLATES = {
    "explain_plus_culture": {
        "slots": {
            "SUBJECT": ["Technology", "Science"],
            "CULTURE": ["CulturalPractice"],
            "LOCATION": ["Location.Region", "Location.Country"]
        },
        "text": (
            "Write about {SUBJECT} - what it is, how it works, and typical uses. "
            "Then separately write about {CULTURE} in {LOCATION}, covering traditions, variations, and social context."
        ),
    },
    "compare_and_user_impact": {
        "slots": {
            "A": ["Technology", "Science", "EconomicConcept"],
            "B": ["Technology", "Science", "EconomicConcept"],
            "IMPACT": ["ImpactDomain"],
            "GROUP": ["PopulationGroup"]
        },
        "text": (
            "Compare {A} and {B}, focusing on principles, performance, and trade-offs. "
            "Then explain how these differences affect {IMPACT} for {GROUP}."
        ),
    },
    "apply_to_sector_plus_policy": {
        "slots": {
            "CONCEPT": ["Technology", "Science"],
            "SECTOR": ["Sector"],
            "POLICY": ["PolicyInstrument"]
        },
        "text": (
            "Write about how {CONCEPT} applies to {SECTOR}, including workflow changes, benefits, and risks. "
            "Then discuss policy measures such as {POLICY} that shape adoption."
        ),
    },
    "cause_effect_mitigation": {
        "slots": {
            "ISSUE": ["EnvironmentIssue", "HealthCondition", "EthicalIssue"],
            "IMPACT": ["ImpactDomain"],
            "LOCATION": ["Location.Country", "Location.City", "Location.Region", "Location.RuralArea"]
        },
        "text": (
            "Describe how {ISSUE} affects {IMPACT} in {LOCATION}, covering mechanisms and evidence. "
            "Then propose practical mitigation strategies and trade-offs."
        ),
    },
    "debate_roles_summary": {
        "slots": {
            "ROLE_A": ["ActorRole"],
            "ROLE_B": ["ActorRole"],
            "TOPIC": ["EthicalIssue", "Technology", "PolicyInstrument"],
            "IMPACT": ["ImpactDomain"]
        },
        "text": (
            "Create a dialogue between a {ROLE_A} and a {ROLE_B} discussing {TOPIC}. Start with the {ROLE_A}'s perspective. "
            "End with a summary addressing {IMPACT}."
        ),
    },
    "timeline_discovery_to_ethics": {
        "slots": {
            "DISCOVERY": ["Science"],
            "SECTOR": ["Sector"],
            "ETHICS": ["EthicalIssue"]
        },
        "text": (
            "Trace the scientific development of {DISCOVERY}. "
            "Then explain its commercialization in {SECTOR}. "
            "Finally, analyze ethical concerns focusing on {ETHICS}."
        ),
    },
    "narrative_plus_technical_anchor": {
        "slots": {
            "EVENT_GROUP": ["PopulationGroup"],
            "TECH": ["Technology"],
            "LOCATION": ["Location.Country", "Location.City", "Location.Region", "Location.RuralArea"]
        },
        "text": (
            "Write a short narrative about {EVENT_GROUP} in {LOCATION}. "
            "Then add a factual explanation of how {TECH} enabled or shaped this experience."
        ),
    },
    "policy_brief_plus_enforcement": {
        "slots": {
            "POLICY": ["PolicyInstrument"],
            "SYSTEM": ["Technology"],
            "LOCATION": ["Location.Country", "Location.City"]
        },
        "text": (
            "Draft a policy brief addressing {POLICY} for {SYSTEM} in {LOCATION}. "
            "Then detail technical enforcement and auditability mechanisms."
        ),
    },
    "governance_plus_algorithm_transparency": {
        "slots": {
            "PLATFORM": ["Technology"],
            "ETHICS": ["EthicalIssue"],
            "IMPACT": ["ImpactDomain"]
        },
        "text": (
            "Outline governance principles for {PLATFORM}, accounting for {ETHICS}. "
            "Then explain how transparency in ranking and recommendation algorithms supports {IMPACT}."
        ),
    },
    "how_to_plus_unrelated_sidebar": {
        "slots": {
            "WORKFLOW": ["Sector", "EducationTheme"],
            "SIDEBAR_LOCATION": ["Location.Country", "Location.Region", "Location.City", "Location.RuralArea"],
            "SIDEBAR_CULTURE": ["CulturalPractice"]
        },
        "text": (
            "Provide a concise how-to guide for {WORKFLOW}. "
            "Then add a sidebar describing {SIDEBAR_CULTURE} in {SIDEBAR_LOCATION}."
        ),
    },
    "data_system_plus_art_reflection": {
        "slots": {
            "SYSTEM": ["Technology"],
            "ART": ["ArtForm"]
        },
        "text": (
            "Explain how {SYSTEM} processes inputs and produces outputs. "
            "Then reflect on parallels with {ART}, considering structure, style, and interpretation."
        ),
    },
    "clinical_vs_public_health_plus_norms": {
        "slots": {
            "CLINICAL": ["HealthCondition"],
            "PUBLIC_HEALTH": ["PolicyInstrument"],
            "COMMUNITY": ["Location.City", "Location.Region", "Location.Country"]
        },
        "text": (
            "Contrast clinical approaches to {CLINICAL} with population-level interventions like {PUBLIC_HEALTH}. "
            "Then discuss how local norms in {COMMUNITY} shape outcomes."
        ),
    },
}

# ----------------------------
# Helpers: sampling, rendering, validation
# ----------------------------
def _flatten_slot_types(slot_types: List[str]) -> List[str]:
    pool = []
    for t in slot_types:
        if t.startswith("Location."):
            branch = t.split(".", 1)[1]
            pool.extend(ONTOLOGY["Location"][branch])
        else:
            pool.extend(ONTOLOGY[t])
    return pool

def get_top_level_domain(slot_types: List[str], selected_item: str) -> str:
    """Determine which top-level domain the selected item came from."""
    for slot_type in slot_types:
        if slot_type.startswith("Location."):
            # For Location subtypes, check the specific branch
            branch = slot_type.split(".", 1)[1]
            if selected_item in ONTOLOGY["Location"][branch]:
                return "Location"
        else:
            # For other types, check directly
            if selected_item in ONTOLOGY[slot_type]:
                return slot_type
    return "Unknown"

def choose_item_for_slot(slot_types: List[str]) -> str:
    candidates = _flatten_slot_types(slot_types)
    if not candidates:
        raise ValueError(f"No items for slot types: {slot_types}")
    return random.choice(candidates)

def render_template(template_id: str, values: Dict[str, str] = None) -> Tuple[str, Dict[str, str]]:
    spec = TEMPLATES[template_id]
    if values is None:
        values = {}
        for slot_name, slot_types in spec["slots"].items():
            values[slot_name] = choose_item_for_slot(slot_types)
    text = spec["text"].format(**values)
    return text, values

def hash_prompt(text: str, values: Dict[str, str]) -> str:
    m = hashlib.sha256()
    m.update(text.encode("utf-8"))
    m.update(json.dumps(values, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

def validate_multi_domain(spec_slots: Dict[str, List[str]]) -> bool:
    # Ensure at least two distinct top-level domains across slots
    domains = set()
    for slot_types in spec_slots.values():
        base = slot_types[0].split(".", 1)[0]  # Location.City -> Location
        domains.add(base)
    return len(domains) >= 2

def lexical_variants(text: str) -> List[str]:
    # Rule-based light paraphrases (pure code) for variation without LLMs
    variants = [text]
    variants.append(text.replace("Paragraph", "Section"))
    variants.append(text.replace("Explain", "Describe"))
    variants.append(text.replace("Discuss", "Analyze"))
    return list(set(variants))

# ----------------------------
# Sankey data tracking
# ----------------------------
class SankeyTracker:
    """Tracks connections for Sankey diagram generation."""
    
    def __init__(self):
        # Layer 1 -> Layer 2: Template -> Slot
        self.template_to_slot = defaultdict(int)
        
        # Layer 2 -> Layer 3: Slot -> Domain
        self.slot_to_domain = defaultdict(int)
        
        # Layer 3 -> Layer 4: Domain -> Item
        self.domain_to_item = defaultdict(int)
        
    def record_prompt(self, template_id: str, values: Dict[str, str]):
        """Record a single prompt's contribution to the Sankey data."""
        spec = TEMPLATES[template_id]
        
        for slot_name, slot_types in spec["slots"].items():
            selected_item = values[slot_name]
            
            # Determine top-level domain
            domain = get_top_level_domain(slot_types, selected_item)
            
            # Create composite keys for clarity
            template_slot_key = f"{template_id}:{slot_name}"
            slot_domain_key = f"{template_slot_key} -> {domain}"
            domain_item_key = f"{domain} -> {selected_item}"
            
            # Increment counts
            self.template_to_slot[f"{template_id} -> {template_slot_key}"] += 1
            self.slot_to_domain[slot_domain_key] += 1
            self.domain_to_item[domain_item_key] += 1
    
    def get_sankey_data(self) -> Dict:
        """Return all Sankey connection data in a structured format."""
        return {
            "layer_1_to_2": dict(self.template_to_slot),
            "layer_2_to_3": dict(self.slot_to_domain),
            "layer_3_to_4": dict(self.domain_to_item),
            "summary": {
                "total_template_slot_connections": len(self.template_to_slot),
                "total_slot_domain_connections": len(self.slot_to_domain),
                "total_domain_item_connections": len(self.domain_to_item)
            }
        }
    
    def export_for_sankey(self, filename: str = "sankey_data.json"):
        """Export data in a format ready for Sankey visualization."""
        data = self.get_sankey_data()
        
        # Create a more structured format for Sankey libraries
        nodes = set()
        links = []
        
        # Process Layer 1 -> Layer 2
        for connection, weight in self.template_to_slot.items():
            source, target = connection.split(" -> ")
            nodes.add(source)
            nodes.add(target)
            links.append({"source": source, "target": target, "value": weight})
        
        # Process Layer 2 -> Layer 3
        for connection, weight in self.slot_to_domain.items():
            source, target = connection.split(" -> ")
            nodes.add(source)
            nodes.add(target)
            links.append({"source": source, "target": target, "value": weight})
        
        # Process Layer 3 -> Layer 4
        for connection, weight in self.domain_to_item.items():
            source, target = connection.split(" -> ")
            nodes.add(source)
            nodes.add(target)
            links.append({"source": source, "target": target, "value": weight})
        
        sankey_format = {
            "nodes": [{"id": node} for node in sorted(nodes)],
            "links": links,
            "raw_data": data
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sankey_format, f, indent=2)
        
        return sankey_format

# ----------------------------
# Generation modes
# ----------------------------
def generate_prompts_pure_code(n: int, seed: int = 42, track_sankey: bool = True) -> Tuple[List[Dict], SankeyTracker]:
    random.seed(seed)
    prompts = []
    seen_hashes = set()
    sankey = SankeyTracker() if track_sankey else None
    
    template_ids = list(TEMPLATES.keys())
    for _ in range(n * 10):  # Generate more attempts to ensure we get n prompts
        tid = random.choice(template_ids)
        spec = TEMPLATES[tid]
        if not validate_multi_domain(spec["slots"]):
            continue
        text, values = render_template(tid)
        for variant in lexical_variants(text):
            record = {
                "template_id": tid,
                "text": variant,
                "values": values,
                "mode": "pure_code"
            }
            h = hash_prompt(variant, values)
            if h not in seen_hashes:
                prompts.append(record)
                seen_hashes.add(h)
                
                # Track for Sankey
                if sankey:
                    sankey.record_prompt(tid, values)
                
            if len(prompts) >= n:
                break
        if len(prompts) >= n:
            break
    return prompts[:n], sankey

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Generate prompts and track Sankey data
    num_prompts = 500
    output_file = os.path.join(os.path.dirname(__file__), "prompts.jsonl")
    sankey_file = os.path.join(os.path.dirname(__file__), "sankey_data.json")
    
    prompts, sankey_tracker = generate_prompts_pure_code(num_prompts, track_sankey=True)
    
    # Write prompts to JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    
    # Export Sankey data
    sankey_data = sankey_tracker.export_for_sankey(sankey_file)
    
    print(f"Generated {len(prompts)} prompts and saved to {output_file}")
    print(f"Sankey data saved to {sankey_file}")
    print("\nSankey Summary:")
    print(f"  - Template->Slot connections: {sankey_data['raw_data']['summary']['total_template_slot_connections']}")
    print(f"  - Slot->Domain connections: {sankey_data['raw_data']['summary']['total_slot_domain_connections']}")
    print(f"  - Domain->Item connections: {sankey_data['raw_data']['summary']['total_domain_item_connections']}")
    print(f"  - Total nodes: {len(sankey_data['nodes'])}")
    print(f"  - Total links: {len(sankey_data['links'])}")
    
    print("\nSample prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"\n{i}. Template: {prompt['template_id']}")
        print(f"   Text: {prompt['text'][:100]}...")
        print(f"   Values: {prompt['values']}")
    
    # Show sample Sankey connections
    print("\n\nSample Sankey Connections:")
    print("\nLayer 1->2 (Template->Slot) - First 5:")
    for i, (conn, weight) in enumerate(list(sankey_data['raw_data']['layer_1_to_2'].items())[:5]):
        print(f"  {conn}: {weight}")
    
    print("\nLayer 2->3 (Slot->Domain) - First 5:")
    for i, (conn, weight) in enumerate(list(sankey_data['raw_data']['layer_2_to_3'].items())[:5]):
        print(f"  {conn}: {weight}")
    
    print("\nLayer 3->4 (Domain->Item) - First 5:")
    for i, (conn, weight) in enumerate(list(sankey_data['raw_data']['layer_3_to_4'].items())[:5]):
        print(f"  {conn}: {weight}")