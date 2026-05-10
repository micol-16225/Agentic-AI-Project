import requests
import os
import time
from bs4 import BeautifulSoup
import re

# --- 1. Configuration ---
# Create a directory to save the downloaded text files
OUTPUT_DIR = 'warning_letters_text'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. The URL List (Using your full list) ---
# NOTE: This list should be updated to contain all 113 URLs you are processing.
URL_LIST = [
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/dermarite-industries-llc-710997-10272025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sinsin-pharmaceutical-co-ltd-718720-11122025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/verdure-sciences-inc-717259-11102025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/melcare-biomedical-pty-ltd-717968-10092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/health-and-natural-beauty-usa-corp-700187-07282025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/dixon-investments-inc-dba-ari-710087-09172025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/pamela-k-den-besten-dds-ms-715245-09192025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/ralph-defronzo-md-716773-09172025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/novo-nordisk-inc-716495-09092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/eli-lilly-and-company-716485-09092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/everlaan-organics-inc-dba-maple-organics-707956-08112025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/scope-health-inc-695085-07092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/thea-pharma-inc-712416-07092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/revitalize-energy-inc-712417-07092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/eyetech-one-llc-712438-07092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/covalent-medical-695839-07092025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/dynamic-blending-specialists-inc-701278-06182025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/amish-origins-management-llc-704166-05292025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/nana-barseghian-md-708009-04282025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cosco-international-inc-701209-04142025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/pms4pms-llc-700180-04072025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/bpi-labs-llc-699533-03202025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/americo-f-padilla-md-700447-03252025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/mehran-michael-bahrami-md-703689-03052025",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/prime-vitality-inc-dba-prime-peptides-695156-12102024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/xcel-research-llc-694608-12102024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/veronvy-694688-12102024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/applied-therapeutics-inc-696833-12032024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lcc-ltd-688576-11212024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/multaler-et-cie-sas-685009-11052024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/choice-all-natural-inc-dba-om-botanical-685615-10212024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/root-bioscience-brands-llc-dba-naternal-673646-08302024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/aqualex-co-ltd-672292-06122024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/ward-smelling-salts-672166-04302024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/kevin-r-bender-mddbc-research-corporation-680072-05022024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/bret-r-rutherford-md-670544-03212024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/ammonia-sport-inc-672196-03192024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/skull-smash-llc-672714-03132024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/spirochaete-research-labs-llc-aka-scitus-laboratory-products-674278-03082024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/innovative-formulations-llc-dba-insane-labz-672952-02222024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/native-salts-llc-674277-02282024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/salvacion-usa-inc-672252-02132024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/us-chem-labs-669074-02072024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/global-life-technologies-corp-674279-02062024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sambrosa-care-inc-673321-01242024",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/novid-group-inc-672251-12072023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/discover-health-llc-dba-discover-cbd-and-strain-snobs-661488-11162023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/handock-cosmetics-co-ltd-658737-10042023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/antonio-e-blanco-mdvista-health-research-llc-668519-09262023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/shot-joy-llc-665936-09252023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/dr-vitamins-llc-dba-dr-vitamin-solutions-663127-09112023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/thrasio-llc-dba-zymaderm-649788-08182023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/angela-m-stupi-md-665471-08082023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/suprimo-imports-657631-06222023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/randy-j-epstein-mdchicago-cornea-consultants-ltd-658917-06142023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/nose-slap-llc-654566-04242023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/medolife-rx-dba-medolife-corp-aelia-inc-and-quanta-inc-651522-04192023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/robert-j-hayashi-mdwashington-university-school-medicine-department-pediatrics-654873-03142023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/maggie-jeffries-md-avanti-anesthesiology-llc-646498-03032023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/larkin-community-hospital-institutional-review-board-638146-01112023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/his-enterprise-inc-dba-adams-secret-usa-llc-646494-01102023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/medical-mikes-inc-647603-01102023",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/thriftmaster-texas-llc-dba-thriftmaster-global-holdings-inc-and-tm-global-biosciences-llc-641057",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/alternative-health-distribution-llc-dba-cannaaid-641241-11012022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/atlrx-inc-618341-05042022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/kingdom-harvest-625058-05042022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/biomd-plus-llc-618460-05042022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/m-six-labs-inc-618701-05042022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/daniel-c-tarquinio-docenter-rare-neurological-diseases-630976-04052022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/wwwwellerectilecom-615690-04062022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sensory-cloud-inc-628897-04062022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sabine-s-hazan-md-628110-02282022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/ivermectin24hcom-615637-02252022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/extrapharmacyru-615132-02152022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/pharmacy2homelandicom-holding-ltd-615137-02032022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/richard-j-obiso-phd-dba-avila-herbals-llc-622821-02042022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cosmo-bio-co-ltd-611552-01062022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/amcyte-pharma-inc-623474-01032022",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/varigard-llc-616757-12292021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lauren-r-klein-md-ms-605544-05062021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/jon-b-cole-md-611902-05052021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/raas-nutritionals-llcrare-antibody-antigen-supply-inc-616560-09302021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/kaleido-biosciences-inc-616026-08262021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/mb-solutions-llcbiospectrum-cbd-610649-07222021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/dream-pharmacy-247-enterprises-limited-2018-614898-07012021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/halodine-llc-611150-07072021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/pharmacygeoffmd-614055-04152021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cannafyl-611957-03012021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/evolved-ayurvedic-discoveries-incbiocbdplus-609772-02112021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/rx2gocom-612391-01282021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sandra-pharmais-611909-01282021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/buy-pharmamd-611907-01282021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/cocos-holistic-specialties-apothecary-612001-01042021",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/nextl3vel-services-group-llc-dba-stuff-good-you-610446-12222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/wellness-biosciences-rx-609604-12222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/novabay-pharmaceuticals-inc-610816-11022020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/extrapharmacom-607931-09022020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/medication-housecom-607751-09022020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/hydroxychloroquine-onlinecom-607727-07092020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/pharmaboosterscom-607929-07092020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/lianhuaqingwencapscom-608667-07062020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/sinotraditioncom-608643-07062020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/center-wellness-and-integrative-medicine-608693-06302020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/4nrxmd-606115-05132020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/roidsmallnet-606133-04222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/emedkitcom-606138-04222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/fusion-health-and-vitality-llc-607545-05112020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/plum-dragon-herbs-inc-05082020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/honey-colony-llc-607346-05042020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/antroidscom-606116-04222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/foxroidscom-606131-04222020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/homero-corp-dba-natures-cbd-oil-distribution-605222-04202020",
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/zhuhai-aofute-medical-technology-co-ltd-590945-01092020"
]

# --- 3. Boilerplate Keywords to Filter Out ---
# These phrases are almost always part of the FDA website template, not the letter body.
BOILERPLATE_KEYS = [
    "About Warning and Close-Out Letters",
    "Content current as of:",
    "Regulated Product(s)",
    "Download Section",
    "How to Contact FDA",
    "Follow FDA",
    "Back to top",
    "Inspections, Compliance, Enforcement, and Criminal Investigations"
]

# --- 4. Scraper Function (The final attempt) ---

# --- 4. Scraper Function (FINAL & HOTFIXED VERSION) ---

def download_and_save_letter(url, index):
    """Fetches the webpage, extracts text by paragraph isolation and filtering, and saves it."""
    print(f"[{index}/{len(URL_LIST)}] Fetching: {url}...")
    
    # Boilerplate Keys are defined outside this function and assumed to be correct
    # BOILERPLATE_KEYS = [...]
    
    try:
        # 1. Setup Request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, headers=headers, timeout=20)
        page.raise_for_status() 

        soup = BeautifulSoup(page.content, 'html.parser')
        
        # 2. Smart Container Search (EXPANDED FIX)
        
        # Priority 1: The standard, specific FDA main content wrapper
        content_wrapper = soup.find('article', id='main-content') 
        
        if not content_wrapper:
             # Priority 2: Standard template main role
            content_wrapper = soup.find('div', role='main')
            
        if not content_wrapper:
            # Priority 3: Common structural div for main content on older/alternate templates
            # Search for a <div> with common content classes (like 'container' or 'region')
            content_wrapper = soup.find('div', {'class': ['container', 'region region-content']})

        if not content_wrapper:
            # Priority 4: Final, broadest fallback to the body tag
            content_wrapper = soup.find('body')

        if not content_wrapper:
            print(f"[{index}/{len(URL_LIST)}] **FAIL** ❌: Could not find any suitable wrapper element.")
            # We return False here to skip the rest of the extraction logic if no container is found
            return False

        # 3. Aggressive Paragraph Isolation & Filtering (The robust logic)
        # Search for all relevant elements within the selected wrapper
        # Prioritize P tags, but include list items, blockquotes, and headers for structure.
        content_elements = content_wrapper.find_all(['p', 'li', 'blockquote', 'h2', 'h3'])
        
        processed_paragraphs = []
        
        # Define the boilerplate keys again for certainty
        BOILERPLATE_KEYS = [
            "About Warning and Close-Out Letters",
            "Content current as of:",
            "Regulated Product(s)",
            "Download Section",
            "How to Contact FDA",
            "Follow FDA",
            "Back to top",
            "Inspections, Compliance, Enforcement, and Criminal Investigations"
        ]

        # Filter and clean the extracted content
        for element in content_elements:
            text = element.get_text(strip=True)
            
            # Skip short elements (< 50 characters)
            if len(text) < 50:
                continue
                
            # Skip boilerplate text
            is_boilerplate = any(key in text for key in BOILERPLATE_KEYS)
            if is_boilerplate:
                continue
            
            # Heuristic to filter navigation/menu items
            if re.search(r'^\s*(Skip to main content|Home|Search|Menu|Table of Contents|Share$)', text, re.IGNORECASE):
                continue
            
            # Final specific cleaning for addresses/contact lines that get missed but are repetitive
            if re.search(r'(Please notify this office in writing|Sincerely|Contact information for the FDA)', text, re.IGNORECASE):
                continue

            processed_paragraphs.append(text)

        # 4. Reassemble the letter text
        letter_text = '\n\n'.join(processed_paragraphs)

        # --- Check for successful extraction and save ---
        if letter_text and len(letter_text) > 500:
            
            # Final cleaning
            letter_text = re.sub(r'\n\s*\n', '\n\n', letter_text)
            
            # Create a unique filename
            filename_segment = url.split('/')[-1] 
            filename = os.path.join(OUTPUT_DIR, f"{filename_segment}.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(letter_text)
            print(f"[{index}/{len(URL_LIST)}] **SUCCESS** ✅: Saved to {filename}")
            return True
        else:
            print(f"[{index}/{len(URL_LIST)}] **FAIL** ❌: Isolation succeeded but final text was too short (Length: {len(letter_text)} chars).")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[{index}/{len(URL_LIST)}] ERROR: Network or Request failed: {e}")
        return False
    except Exception as e:
        print(f"[{index}/{len(URL_LIST)}] ERROR: General failure: {e}")
        return False
    
# --- 5. Main Execution Loop ---
if __name__ == "__main__":
    # URL_LIST should be defined earlier in your script
    total_urls = len(URL_LIST)
    success_count = 0
    fail_count = 0

    print(f"--- Starting Scraping of {total_urls} URLs ---")
    
    # Iterate through the list, keeping track of the index (i) and the URL
    for i, url in enumerate(URL_LIST):
        # Index for display starts at 1, not 0
        index = i + 1 
        
        # Call the scraping function
        result = download_and_save_letter(url, index)
        
        if result:
            success_count += 1
        else:
            fail_count += 1
        
        # Wait for 3 seconds to be polite to the server and avoid being blocked
        time.sleep(3) 

    print("--- Scraping Complete ---")
    print(f"Total URLs Processed: {total_urls}")
    print(f"Successful Extractions: {success_count} ✅")
    print(f"Failed Extractions: {fail_count} ❌")