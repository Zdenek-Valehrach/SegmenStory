def generate_prompt(object_name, context, iteration=False, previous_text=None):
    """
    Generates prompt templates for different contexts
    
    Args:
        object_name: Name of the detected object
        context: Type of context for the prompt
        iteration: Boolean indicating if this is an alternative generation
        previous_text: Previously generated text (for iterations)
        
    Returns:
        Appropriate prompt text for the given context
    """
    base_prompts = {
        "Behaviorální jednání": f"Popiš behaviorální jednání objektu '{object_name}'.",
        "Evoluční antropologie": f"Popiš evoluční význam objektu. '{object_name}'.",
        "Společenské chování": f"Popiš společenský význam objektu '{object_name}'."
    }
    
    # Základní prompt
    base_prompt = base_prompts.get(context, f"Popiš zajímavé informace o '{object_name}'.")
    
    # Pokud jde o iteraci, přidáme předchozí text a požádáme o alternativní pohled
    if iteration and previous_text:
        return f"""Vygeneruj alternativní text o objektu '{object_name}' z pohledu: {context}.
        
Předchozí popis byl:
{previous_text}

Nabídni jiný, kreativní pohled na stejný objekt. Použij jiné informace, styl nebo perspektivu než v předchozím popisu."""
    
    return base_prompt