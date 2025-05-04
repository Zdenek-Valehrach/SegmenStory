# SegmenStory

Tento projekt vznikl jako moje vlastní *playground aplikace* pro segmentaci obrázků a generování vtipných "evolučních" příběhů k jednotlivým segmentům pomocí LLM. Cílem bylo spojit moderní počítačové vidění (Mask2Former přes Hugging Face API) s generativní AI (Perplexity LLM, OpenAI) a vytvořit jednoduché, ale uživatelsky přívětivé rozhraní ve Streamlitu. Projekt byl vytvořen zejména pro otestování některých řešení pro mé budoucí pokročilejší projekty.  

## Co aplikace umí

- **Segmentace obrázků**: Po nahrání obrázku je pomocí Mask2Former modelu detekováno, jaké objekty na obrázku jsou.
- **Vizualizace segmentů**: Barevně označené segmenty přímo na obrázku pomocí průhledných masek.
- **Výběr třídy**: Z rozpoznaných tříd si můžeš jednoduše vybrat konkrétní objekt (segment).
- **Generování příběhu**: Možnost volby mezi Perplexity LLM a OpenAI pro generování vtipných evolučních příběhů.
- **Překlady tříd do češtiny**: Všechny třídy jsou uživateli zobrazeny v češtině díky vlastnímu slovníku, což zvyšuje srozumitelnost a konzistenci.
- **Streamlit UI**: Intuitivní rozhraní s možností resetu, jasně komunikovanými stavy aplikace a uživatelskými volbami.
- **Pole pro zadání API klíčů**: Uživatel může zadat vlastní API klíče pro Hugging Face, Perplexity i OpenAI přímo v aplikaci.
- **Přepínání mezi Perplexity a OpenAI**: Uživatel si může vybrat, který model bude generovat příběh.
- **Možnost vložit vlastní téma**: Lze zadat libovolné téma pro generování příběhu, nezávisle na segmentaci obrázku.


## Jak jsem postupoval

1. **Základní architektura**:
Rozdělil jsem projekt na moduly pro segmentaci (`segmentation.py`), zpracování obrázků (`image_utils.py`), generování textu (`llm.py`), tvorbu promptů (`prompt_utils.py`) a překlad tříd (`coco_class_map.py`).
2. **Integrace Mask2Former přes Hugging Face API**:
Pro segmentaci obrázků jsem použil API, což znamenalo nutnost zmenšovat obrázky (kvůli limitům payloadu) a řešit převod formátů.
3. **Vykreslení segmentačních masek**:
Implementoval jsem systém pro dekódování a zobrazení masek, které vrací Mask2Former API. Původně se mi zobrazoval jen seznam tříd, ale nyní jsou segmenty barevně zvýrazněny přímo na obrázku. Pro toto jsem musel:
    - Dekódovat base64 zakódované masky z API odpovědi
    - Vytvořit systém generování vizuálně odlišných barev pro jednotlivé segmenty pomocí HSV barevného modelu
    - Implementovat aplikaci poloprůhledných barevných vrstev na původní obrázek
    - Vyřešit konverzi mezi různými formáty obrázků (numpy array, PIL Image, RGBA/RGB)
4. **Barevné kódování segmentů**:
Pro lepší vizuální odlišení různých objektů jsem naprogramoval funkci `generate_distinct_colors`, která rovnoměrně rozděluje barevné spektrum podle počtu detekovaných segmentů. Díky tomu každý objekt dostane jedinečnou barvu, což výrazně zlepšuje orientaci v segmentovaném obrázku.
5. **Perplexity LLM a OpenAI API**:
Pro generování příběhů jsem nejprve integroval Perplexity API (model `sonar`), následně jsem přidal možnost volání API od OpenAI (model `gpt-4o`). Uživatel si může zvolit, který model použije.
6. **Pole pro zadání API klíčů**:
V aplikaci jsou na sidebaru pole pro zadání vlastních API klíčů pro všechny podporované služby. Pokud uživatel nevyplní klíč, použije se výchozí z configu.
7. **Přepínání mezi Perplexity a OpenAI**:
Přidal jsem přepínač na sidebaru, kterým lze zvolit poskytovatele generování textu.
8. **Možnost vložit vlastní téma**:
Přidal jsem textové pole, kde lze zadat libovolné téma pro generování příběhu, nezávisle na segmentaci obrázku.
9. **Překlady tříd a robustní mapování**:
Mask2Former vrací třídy i v různých variantách (např. `tree-merged`, `wall-other-merged`, nebo i zcela nové jako `snow`). Proto jsem vytvořil slovník a funkci na předzpracování názvů, která zvládne i neočekávané případy.
10. **Streamlit UI a UX**:
Přidal jsem progress bar pro zábavnější čekání na výsledek a ošetřil případy, kdy segmentace selže nebo nevrátí žádné objekty – uživatel dostane jasnou hlášku, co dělat dál.
11. **Odstraňování artefaktů z LLM výstupu**:
Perplexity někdy generuje zbytečné číselné odkazy `[^1]`, které jsem pomocí regulárních výrazů odstranil, ale zároveň zachoval případné skutečné citace.

### Srovnání modelů pro generování vtipného textu

Oba modely mají stejné nastavení role/kontext, temperature, atd. Na základě mého testování platí:

- **OpenAI (gpt-4o):**
    - Výsledky jsou konzistentní, gramaticky správné, ale často méně kreativní. Úvodní části textů jsou mnohdy velmi podobné.
    - Model má tendenci držet se „bezpečných“ a ověřených vzorců (což je dáno i RLHF tréninkem a zaměřením na bezpečnost a univerzálnost odpovědí).
    - Vtipnost a originalita výstupu nejsou vždy na špičkové úrovni, pokud není prompt výrazně kreativní nebo není speciálně upravená role/kontext. 
- **Perplexity (sonar):**
    - Výstupy jsou nápaditější, často vtipnější, s větší variabilitou a překvapením v textu.
    - Model je méně konzervativní, ochotnější „riskovat“ s neotřelými formulacemi a humorem.
    - Občas udělá drobnou gramatickou chybu, ale rozdíl je minimální.

Tento rozdíl je dán jak tréninkovými daty, tak optimalizací modelu – OpenAI modely jsou často laděny na univerzální bezpečné použití, zatímco Perplexity se nebojí větší kreativity a „odvázanosti“ v odpovědích. Pokud bychom modelu od OpenAI přidali i příklad výstupu z Perplexity, výsledky by byly určitě lepší. 

## Co jsem musel vyřešit

- **Správné importy a cesty v projektu** – hlavně při rozdělení do více složek a spouštění v různých prostředích.
- **Zpracování masek z Hugging Face API** – dekódování base64 řetězců na obrázky a jejich správné překrytí přes originál.
- **Generování barevného schématu** – vytvoření algoritmu pro generování vizuálně odlišných barev podle počtu segmentů.
- **Zpracování a překlad tříd** – aby byly všechny objekty vždy česky a uživatel nebyl zmatený anglickým názvem nebo neznámou třídou.
- **Konverze formátů obrázků** – správná manipulace s numpy arrays, PIL Images a konverze mezi RGB a RGBA pro transparentní překrytí.
- **Ošetření neočekávaných výstupů segmentace** – například nové nebo sloučené třídy, které nebyly v původním slovníku.
- **Ošetření selhání segmentace** – aby uživatel dostal jasné pokyny, když segmentace nic nevrátí.
- **Limity API (velikost obrázku, počet tokenů, formát promptu)** – bylo nutné optimalizovat velikost obrázků a promptů, aby vše fungovalo spolehlivě a odpovědi byly kompletní.
- **Odstranění artefaktů z LLM výstupu** – např. `[1]`, které Perplexity někdy přidává bez významu.


## Instalace a spuštění

1. Nainstaluj závislosti:

```
pip install -r requirements.txt
```

2. Zajisti si API klíče pro Hugging Face, Perplexity nebo OpenAI a vlož je do `config.py` (nebo je zadej přímo v aplikaci do příslušných polí v sidebaru).
3. Spusť aplikaci:

```
streamlit run app.py
```


## Další poznámky

- **API klíče**: Pro testování a bezpečný provoz doporučuji zadávat API klíče přímo v aplikaci. Pokud pole zůstane prázdné, použije se klíč z `config.py` (pokud je k dispozici).
- **Volba modelu**: Uživatel si může jednoduše přepnout mezi Perplexity a OpenAI pro generování příběhu.
- **Vlastní téma**: Mimo segmentované objekty lze vygenerovat příběh i pro libovolné uživatelské téma.
