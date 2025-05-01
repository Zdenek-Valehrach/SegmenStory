# SegmenStory

Tento projekt vznikl jako moje vlastní playground aplikace pro segmentaci obrázků a generování vtipných "evolučních" příběhů k jednotlivým segmentům pomocí LLM. Cílem bylo spojit moderní počítačové vidění (Mask2Former přes Hugging Face API) s generativní AI (Perplexity LLM) a vytvořit jednoduché, ale uživatelsky přívětivé rozhraní ve Streamlitu.

## Co aplikace umí

- **Segmentace obrázků**: Po nahrání obrázku je pomocí Mask2Former modelu detekováno, jaké objekty na obrázku jsou.
- **Výběr třídy**: Z rozpoznaných tříd si mohu v aplikaci jednoduše vybrat konkrétní objekt (segment).
- **Generování příběhu**: Po výběru třídy vygeneruje Perplexity LLM vtipný, evolučně-antropologický příběh o vybraném objektu.
- **Překlady tříd do češtiny**: Všechny třídy jsou uživateli zobrazeny v češtině díky vlastnímu slovníku, což zvyšuje srozumitelnost a konzistenci.
- **Streamlit UI**: Intuitivní rozhraní s možností resetu a jasně komunikovanými stavy aplikace.

## Jak jsem postupoval

1. **Základní architektura**:  
   Rozdělil jsem projekt na moduly pro segmentaci (`segmentation.py`), zpracování obrázků (`image_utils.py`), generování textu (`llm.py`), tvorbu promptů (`prompt_utils.py`) a překlad tříd (`coco_class_map.py`).

2. **Integrace Mask2Former přes Hugging Face API**:  
   Pro segmentaci obrázků jsem použil API, což znamenalo nutnost zmenšovat obrázky (kvůli limitům payloadu) a řešit převod formátů.

3. **Perplexity LLM API**:  
   Pro generování příběhů jsem napojil Perplexity API s modelem `sonar`. Musel jsem řešit správné nastavení promptu, aby odpovědi byly v češtině, vtipné a zároveň se vešly do limitu tokenů.

4. **Překlady tříd a robustní mapování**:  
   Narazil jsem na problém, že Mask2Former vrací třídy i v různých variantách (např. `tree-merged`, `wall-other-merged`, nebo i zcela nové jako `snow`). Proto jsem vytvořil robustní slovník a funkci na předzpracování názvů, která zvládne i neočekávané případy.

5. **Streamlit UI a UX**:  
   Přidal jsem progress bar pro zábavnější čekání na výsledek a ošetřil případy, kdy segmentace selže nebo nevrátí žádné objekty – uživatel dostane jasnou hlášku, co dělat dál.

6. **Odstraňování artefaktů z LLM výstupu**:  
   Musel jsem řešit, že Perplexity někdy generuje zbytečné číselné odkazy `[1]`, které jsem pomocí regulárních výrazů odstranil, ale zároveň zachoval případné skutečné citace.

## Co jsem musel vyřešit

- **Správné importy a cesty v projektu** – hlavně při rozdělení do více složek a spouštění v různých prostředích.
- **Zpracování a překlad tříd** – aby byly všechny objekty vždy česky a uživatel nebyl zmatený anglickým názvem nebo neznámou třídou.
- **Ošetření neočekávaných výstupů segmentace** – například nové nebo sloučené třídy, které nebyly v původním slovníku.
- **Ošetření selhání segmentace** – aby uživatel dostal jasné pokyny, když segmentace nic nevrátí.
- **Limity API (velikost obrázku, počet tokenů, formát promptu)** – bylo nutné optimalizovat velikost obrázků a promptů, aby vše fungovalo spolehlivě a odpovědi byly kompletní.
- **Odstranění artefaktů z LLM výstupu** – např. [1], [2], které Perplexity někdy přidává bez významu.

## Instalace a spuštění

1. Nainstaluj závislosti:

pip install -r requirements.txt

2. Zajisti si API klíče pro Hugging Face i Perplexity a vlož je do `config.py`.
3. Spusť aplikaci:

streamlit run app.py

Na projektu ještě pracuji!