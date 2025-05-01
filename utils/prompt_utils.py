class PromptBuilder:
    """
    Pomocná třída pro tvorbu promptů pro LLM.
    """
    def __init__(self, system_role=None, context=None):
        self.system_role = system_role or (
            "Jsi největší odborník na evoluční antropologii s neodolatelným smyslem pro humor. "
            "Tvé znalosti sahají od prehistorických nástrojů po moderní technologie a vždy dokážeš vykouzlit smích na tváři."
        )
        self.context = context or (
            "Popiš evoluční příběh zadaného objektu, vysvětli jeho vznik jako produkt lidské evoluce a adaptace, "
            "Popiš jeho (možný) vliv na historii lidstva a jakou roli by mohl hrát v budoucnosti. "
            "Vytvoř vtipný příběh a zaměř se na evoluční adaptace, které vedly k jeho vzniku a významu."
            "Ať je příběh vtipný a maximálně na půl stránky."
        )
        self.examples = []

    def add_example(self, user_input, expected_output):
        """
        Přidá příklad (pro few-shot promptování).
        """
        self.examples.append({
            "input": user_input,
            "output": expected_output
        })

    def build(self, user_input):
        """
        Vytvoří finální prompt pro LLM.
        """
        prompt = f"Systémová role: {self.system_role}\n"
        if self.context:
            prompt += f"Kontext: {self.context}\n"
        if self.examples:
            prompt += "Příklady:\n"
            for ex in self.examples:
                prompt += f"Uživatel: {ex['input']}\nAsistent: {ex['output']}\n"
        prompt += f"Uživatel: {user_input}\nAsistent:"
        return prompt



