import re

def extract_pragmatics(text):
    """
    Separates pragmatic signals from the base linguistic content using deterministic rules.
    
    Args:
        text (str): The raw Romanized/code-mixed input string.
        
    Returns:
        dict: A dictionary containing:
            - "content_after_pragmatic_extraction" (str): The text with pragmatic noise extracted.
            - "pragmatic_annotations" (dict): Grouped tags.
            - "trace" (list): Structured modifications for full traceability/reconstruction.
    """
    annotations = {"emojis": [], "punctuation_clusters": [], "repetition": []}
    trace = []
    
    # We will process string modifications in reverse order so that indices remain stable
    # Let's collect all matches first.
    modifications = []
    
    # 1. Emojis
    # We will use a unicode range regex for emojis (no external dependencies).
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27bf]')
    for match in emoji_pattern.finditer(text):
        modifications.append({
            "type": "emoji",
            "raw": match.group(0),
            "replacement": "",
            "start": match.start(),
            "end": match.end()
        })
        
    # 2. Punctuation Clusters
    punct_pattern = re.compile(r'([!?.]){2,}')
    for match in punct_pattern.finditer(text):
        modifications.append({
            "type": "punctuation_cluster",
            "raw": match.group(0),
            "replacement": " ",
            "start": match.start(),
            "end": match.end()
        })
        
    # 3. Repeated Characters
    # Match 3 or more occurrences of any alphabetic character.
    # E.g., pleaseee -> pleasee, kyaaaaa -> kyaa
    rep_pattern = re.compile(r'([a-zA-Z])\1{2,}')
    for match in rep_pattern.finditer(text):
        modifications.append({
            "type": "repetition",
            "raw": match.group(0),
            "replacement": match.group(1) * 2, # Collapse to exactly 2
            "start": match.start(),
            "end": match.end()
        })

    # Sort modifications by start index descending so we can apply them safely
    # If there are overlaps (e.g. repeated punctuation that is also an emoji, unlikely but possible), 
    # we just need to ensure we don't double replace. We'll skip overlaps.
    modifications.sort(key=lambda x: x["start"], reverse=True)
    
    content = text
    last_start = len(text) + 1
    
    for mod in modifications:
        start = mod["start"]
        end = mod["end"]
        raw = mod["raw"]
        replacement = mod["replacement"]
        
        # Skip overlapping modifications
        if end > last_start:
            continue
            
        # Verify safety (sanity check)
        if content[start:end] == raw:
            content = content[:start] + replacement + content[end:]
            last_start = start
            
            # Record in trace
            trace.append({
                "type": mod["type"],
                "raw": raw,
                "replacement": replacement,
                "start": start,
                "end": end
            })
            
            # Record in annotations
            if mod["type"] == "emoji":
                annotations["emojis"].append(raw)
            elif mod["type"] == "punctuation_cluster":
                annotations["punctuation_clusters"].append(raw)
            elif mod["type"] == "repetition":
                annotations["repetition"].append({"raw": raw, "replacement": replacement})
                
    # Trace list might be more natural in forward order
    trace.reverse()

    return {
        "content_after_pragmatic_extraction": content,
        "pragmatic_annotations": annotations,
        "trace": trace
    }
