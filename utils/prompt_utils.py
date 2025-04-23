def generate_prompt(object_name, context):
    """
    Generates prompt templates for different contexts
    
    Args:
        object_name: Name of the detected object
        context: Type of context for the prompt
        
    Returns:
        Appropriate prompt text for the given context
    """
    prompts = {
        "Behaviorální jednání": f"Popiš behaviorální jednání objektu '{object_name}'.",
        "Evoluční antropologie": f"Popiš evoluční význam objektu '{object_name}'.",
        "Společenské chování": f"Popiš společenský význam objektu '{object_name}'."
    }
    
    return prompts.get(context, f"Popiš zajímavé informace o '{object_name}'.")