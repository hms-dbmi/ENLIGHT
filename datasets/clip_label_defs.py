LABEL_2_PROMPT = {
    'AGGC22':{
        "NC": [
            "Non-cancerous",
            "Benign",
            "Normal tissue",
            "Non-malignant",
            "stroma"
        ],
        "G3": [
            "Gleason grade 3",
            "Atrophic well differentiated and dense glandular regions",
            "Low-grade cancer",
            "Well-differentiated glands"
        ],
        "G4": [
            "Gleason grade 4",
            "Cribriform, ill-formed, large-fused and papillary glandular patterns",
            "Intermediate-grade cancer",
            "Moderately differentiated glands"
        ],
        "G5": [
            "Gleason grade 5",
            "Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns",
            "High-grade cancer",
            "Poorly differentiated or undifferentiated cells"
        ]
    },
    'SICAPv2':{
        "NC": [
            "Non-cancerous",
            "Benign",
            "Normal tissue",
            "Non-malignant",
            "stroma"
        ],
        "G3": [
            "Gleason grade 3",
            "Atrophic well differentiated and dense glandular regions",
            "Low-grade cancer",
            "Well-differentiated glands"
        ],
        "G4": [
            "Gleason grade 4",
            "Cribriform, ill-formed, large-fused and papillary glandular patterns",
            "Intermediate-grade cancer",
            "Moderately differentiated glands"
        ],
        "G5": [
            "Gleason grade 5",
            "Isolated cells or file of cells, nests of cells without lumina formation and pseudo-rosetting patterns",
            "High-grade cancer",
            "Poorly differentiated or undifferentiated cells"
        ]
    },
    'RCC-KMC':{
        "0": [
            "grade-0",
            "normal kidney tissue",
        ],
        "1": [
            "grade-1",
            "very low-grade cancer",
        ],
        "2": [
            "grade-2",
            "low-grade cancer",
        ],
        "3": [
            "grade-3",
            "intermediate-grade cancer",
        ],
        "4": [
            "grade-4",
            "high-grade cancer",
        ]
    },
    'BreCaHAD':{
        "tumor": [
            "Mitosis",
            "Cell division",
            "Proliferative activity",
            "Mitotic figures",
            "Active cell division",
            "Tumor nuclei",
            "Cancerous nuclei",
            "Malignant nuclei",
            "Nuclei of cancer cells",
            "Tumor cell nuclei"
        ],
        "non_tumor": [
            "Apoptosis",
            "Programmed cell death",
            "Cellular apoptosis",
            "Apoptotic cells",
            "Cell death",
            "Fibrosis",
            "Non-tumor nuclei",
            "Benign nuclei",
            "Normal cell nuclei",
            "Non-cancerous nuclei",
            "Nuclei of healthy cells",
            "Tubule",
            "Glandular structure",
            "Tubular formation",
            "Tubular gland",
            "Ductal structure",
            "Non-tubule",
            "Non-glandular structure",
            "Non-tubular formation",
            "Absence of tubules",
            "Non-ductal structure"
        ]
    },
    'BIDC':{
        '0': [
            "invasive ductal carcinoma negative",
            "no invasive ductal carcinoma",
            "benign breast tissue",
            "non-cancerous ducts"
        ],
        '1':[
            "invasive ductal carcinoma positive",
            "malignant ductal tumor",
            "cancerous ductal cells",
            "tumor in breast ducts"
        ],
    },
    'HD30000':{
        "LUAD": [
            "luad",
            "lung adenocarcinoma",
            "adenocarcinoma of the lung",
            "gland-forming lung cancer",
            "non-small cell lung adenocarcinoma"
        ],
        "LUSC": [
            "lusc",
            "lung squamous cell carcinoma",
            "squamous cell cancer of the lung",
            "keratinizing lung tumor",
            "non-small cell lung squamous carcinoma"
        ],
        "MESO": [
            "meso",
            "mesothelioma",
            "pleural mesothelioma",
            "mesothelial tumor",
            "cancer of the pleural lining"
        ]
    },
    'Hist700':{
        "aca_bd": [
            "structured adenocarcinoma",
            "low-grade adenocarcinoma",
            "adenocarcinoma showing well differentiation",
        ],
        "aca_md": [
            "moderate adenocarcinoma",
            "intermediate-grade adenocarcinoma",
            "adenocarcinoma showing moderate differentiation",
        ],
        "aca_pd": [
            "aggressive adenocarcinoma",
            "high-grade adenocarcinoma",
            "adenocarcinoma showing poor differentiation",
            
        ],
        "nor": [
            "normal alveolar structure",
            "normal lung tissue",
            "healthy lung tissue",
        ],
        "scc_bd": [
            "organized squamous cancer",
            "organized squamous cell carcinoma",
            "low-grade squamous cell carcinoma",
            "squamous cell carcinoma showing well differentiation"
        ],
        "scc_md": [
            "moderate squamous cancer",
            "moderate squamous cell carcinoma",
            "intermediate-grade squamous cell carcinoma",
            "squamous cell carcinoma showing moderate differentiation"
        ],
        "scc_pd": [
            "aggressive squamous cancer",
            "aggressive cell carcinoma",
            "high-grade squamous cell carcinoma",
            "squamous cell carcinoma showing poor differentiation"
        ]
    },
    'WSSS4LUAD':{
        "tumor": [
            "tumor tissue",
            "cancer tissue",
            "lung adenocarcinoma",
            "adenocarcinoma"],
        "normal": [
            "non-tumor",
            "non-cancerous",
            "benign tissue",
            "normal tissue",
            "healthy tissue",
            "stroma"]
    },
    'NPC-88k':{
        "NORMAL": ["normal tissue", "unremarkable tissue", "unremarkable area"],
        "LHP": ["lymphoid hyperplasia","reactive lymphoid area",],
        "NPI": ['inflammation', "inflammatory changes in nasopharynx", "inflamed nasopharynx",],
        "NPC": ["tumor", "cancered nasopharynx" "malignant nasopharyngeal lesion","npc tumor"]
    },
    'OSCC':{
        'Normal':["normal oral tissue",
            "healthy oral cavity",
            "non-diseased oral mucosa",
            "unaltered oral lining"],
        'OSCC':["oral squamous cell carcinoma",
            "oral cancer",
            "malignant oral tumor",
            "squamous carcinoma of the mouth"]
    },
    'Tolkach':{
        'ADVENT':[
            "adventitial tissue",
            "external lining",
        ],
        'LAM_PROP':[
            "lamina propria mucosa",
            "loose mucosal layer",
        ],
        "MUSC_MUC": [
            "lamina muscular mucosa",
            "mucosal muscle band",
        ],
        "MUSC_PROP": [
            "muscular propria",
            "deep muscle layer",
        ],
        "REGR_TU": [
            "regression area",
            "regressed tumor site",
        ],
        "SH_MAG": [
            "gastric mucosa",
            "mucosa of stomach",
        ],
        "SH_OES": [
            "esophageal mucosa",
            "inner esophageal surface",
        ],
        "SUB_GL": [
            "submucosal glands",
            "subsurface glands",
        ],
        "SUBMUC": [
            "submucosa",
            "underlying mucosa",
        ],
        "ULCUS": [
            "ulceration areas",
            "erosive zone",
        ],
        "TUMOR": [
            "tumor",
            "neoplasm",
            "malignant mass",
            "cancerous growth",
            "tumor area"
        ]
    },
    'DigestPath19_colon':{
        0: [
            "normal colon mucosa",
            "uninvolved colon mucosa",
            "normal colonic mucosa",
            "benign epithelium",
            "healthy colon mucosa",
            "non-pathologic mucosa",
            "normal mucosal tissue",
            "normal colon tissue",
            "normal colonic tissue"
        ],
        1: [
            "metastatic colon adenocarcinoma",
            "metastatic adenocarcinoma",
            "colon adenocarcinoma",
            "colorectal adenocarcinoma",
            "colonic adenocarcinoma",
            "malignant colon tumor",
            "colorectal cancer",
            "colonic cancer",
            "adenocarcinoma of the colon",
            "COAD",
            "high grade intraepithelial neoplasia and adenocarcinoma"
            "papillary adenocarcinoma"
            "mucinous adenocarcinoma"
            "poorly cohesive carcinoma"
            "signet ring cell carcinoma"]
    },
    'Choledoch':{
         "0": [
            "non-cancer",
            "normal tissue",
            "benign area",
            "healthy bile duct"
        ],
        "1": [
            "cancer",
            "malignant tissue",
            "tumor area",
            "cancerous bile duct"
        ]
    },
    'PAIP21':{
        1: [
            "Nerve tissue",
            "Benign nerve tissue",
        ],
        2: [
            "Perineural invasion junction",
            "nerve sheath infiltration by tumor",
        ],
        3: [
            "Tumor without any associated nerve involvement",
            "Malignant tumor without nerve structures",
            "Tumor growth without evidence of nerve infiltration"
        ],
        4: [
            "Tissue without nerve or tumor",
            "Normal or non-tumorous, non-neural tissue",
            "Non-cancerous, non-neural stroma or supportive tissue"
        ]
    },
    'DigestPath19_gastric':{
        "0": [
            "no signet ring cell",
            "absence of signet ring cells",
            "no signet cells seen",
            "signet ring cell not present",
            "signet cell negative"
        ],
        "1": [
            "signet ring cell existing",
            "presence of signet ring cells",
            "signet cells identified",
            "signet ring cell positive",
            "signet cell present"
        ]
    },
    'CAMEL':{
        0: [
            "normal tissue",
            "healthy tissue",
            "unaltered colon",
            "non-neoplastic tissue"
        ],
        1: [
            "colorectal adenoma",
            "benign colon tumor",
            "adenomatous lesion",
            "precancerous adenoma",
            "premalignant adenoma",
        ]
    }, 
    'Chaoyang': {
        "normal": [
            "normal colon mucosa",
            "uninvolved colon mucosa",
            "normal colonic mucosa",
            "benign epithelium",
            "healthy colon mucosa",
            "non-pathologic mucosa",
            "normal mucosal tissue",
            "normal colon tissue",
            "normal colonic tissue"
        ],
        "serrated": [
            "serrated patterns in colon tissue",
            "serrated epithelium",
            "serrated architecture",
            "saw-tooth pattern in colon",
            "serrated crypts",
            "serrated lesions",
            "sessile serrated adenoma",
            "sessile serrated polyp",
            "SSA",
            "SSA/P",
            "sessile serrated lesion",
            "serrated adenoma",
            "pre-malignant serrated polyp",
            "broad-based crypts",
            "complex crypt structure",
            "heavily serrated crypts",
            "complex serrated crypts",
            "crypts with serration",
            "broad crypt architecture",
            "complex crypt architecture",
            "crypts with heavy serration"],
        "adenocarcinoma": [
            "colon adenocarcinoma",
            "colorectal adenocarcinoma",
            "colonic adenocarcinoma",
            "malignant colon tumor",
            "colorectal cancer"],
        "adenoma": [
            "colon adenoma",
            "colorectal adenoma",
            "colonic adenoma",
            "benign colon tumor",
            "adenomatous polyp",
            "adenomatous lesion",
            "colonic polyp",
            "precancerous adenoma",
            "premalignant adenoma",
            "precursor lesions",
            "early-stage cancer lesions",
            "pre-cancerous polyps"]
    },
    'MHIST':{
        'HP': ['Hyperplastic Polyp',
               'superficial serrated architecture and elongated crypts',
               'benign'],
        'SSA': ['Sessile Serrated Adenoma',
                'broad-based crypts, often with complex structure and heavy serration',
                'precancerous lesions']
    },
    'SPIDER_colon':{
        "Sessile serrated lesion": ["serrated lesions","sessile serrated lesion","serrated patterns in colon tissue"],
        "Adenoma high grade": ["high-grade adenoma","adenoma with high-grade dysplasia"], 
        "Adenoma low grade": ["low-grade adenoma", "adenoma with low-grade dysplasia"],
        "Adenocarcinoma high grade": ["aggressive colorectal carcinoma","poorly differentiated adenocarcinoma","undifferentiated colorectal cancer", "high-grade adenocarcinoma",],
        "Adenocarcinoma low grade": ["early-stage colorectal carcinoma","low-grade adenocarcinoma", "well-differentiated adenocarcinoma"],
        "Hyperplastic polyp": ["hyperplastic lesion","non-neoplastic polyp","hyperplastic polyp"],
        "Necrosis": ["necrotic debris","debris","necrosis"],
        "Inflammation": ["inflamed tissue","immune infiltration","inflammatory response","inflammation"],
        "Muscle": ["muscularis mucosa","musculature","muscle"],
        "Fat": ["adipose","adipose tissue","fat cells"],
        "Mucus":  ["mucus pool","mucin pool","mucin","mucus"],
        "Vessels": ["microvessels","capillaries","vasculature","vascular structures","blood vessels","vessels"]
    },
    'UnitToPatho':{
        "NORM": [
            "normal colon mucosa",
            "uninvolved colon mucosa",
            "normal colonic mucosa",
            "benign epithelium",
            "healthy colon mucosa",
            "non-pathologic mucosa",
            "normal mucosal tissue",
            "normal colon tissue",
            "normal colonic tissue"
        ],
        "HP": [
            "colonic polyp",
            "pre-cancerous polyps",
            "Hyperplastic Polyp"],
        "LG": [
            "colon adenoma",
            "colorectal adenoma",
            "colonic adenoma",
            "Tubular Adenoma, Low-Grade dysplasia",
            "Tubulo-Villous Adenoma, Low-Grade dysplasia"
        ],
        "HG": [
            "precancerous adenoma",
            "premalignant adenoma",
            "precursor lesions",
            "pre-cancerous polyps",
            "Tubular Adenoma, High-Grade dysplasia",
            "Tubulo-Villous Adenoma, High-Grade dysplasia",
        ]
    },
    'OCELOT':{
        1: ['Background'],
        2: ['Cancer area', 'tumor cell']
    },
    'Camelyon16': {
        "metastases": [
            "lymph nodes with metastasis"],
        "no metastases": [
            "lymph nodes without metastasis"] 
    },
    'Camelyon17': {
        "metastases": [
            "lymph nodes with metastasis"],
        "no metastases": [
            "lymph nodes without metastasis"] 
    },
    'SkinCancer':{
            "nontumor_skin_chondraltissue_chondraltissue": [
                "Chondral tissue"
            ],
            "nontumor_skin_dermis_dermis": [
                "Dermis",
                "dermal layer",
                "connective tissue layer",
                "middle skin layer"
            ],
            "nontumor_skin_elastosis_elastosis": [
                "Elastosis",
                "elastic degeneration",
                "solar elastosis",
                "elastotic tissue"
            ],
            "nontumor_skin_epidermis_epidermis": [
                "Epidermis",
                "epidermal layer",
                "outer skin layer",
                "surface epithelium",
                "skin surface"
            ],
            "nontumor_skin_hairfollicle_hairfollicle": [
                "Hair follicle",
                "follicle",
                "follicular unit"
            ],
            "nontumor_skin_muscle_skeletal": [
                "Skeletal muscle",
                "voluntary muscle",
                "striated muscle"
            ],
            "nontumor_skin_necrosis_necrosis": [
                "Non-tumor necrosis",
                "Non-tumor necrotic debris"
            ],
            "nontumor_skin_nerves_nerves": [
                "Peripheral nerves",
                "nerve fibers",
                "neural tissue",
                "peripheral neural structures"
            ],
            "nontumor_skin_sebaceousglands_sebaceousglands": [
                "Sebaceous glands",
                "glandular structures"
            ],
            "nontumor_skin_subcutis_subcutis": [
                "Subcutis",
                "subcutaneous tissue",
                "fatty layer"
            ],
            "nontumor_skin_sweatglands_sweatglands": [
                "Sweat glands",
                "eccrine glands",
                "apocrine glands",
                "sweat ducts",
                "skin glands"
            ],
            "nontumor_skin_vessel_vessel": [
                "Blood vessels",
                "vessels",
                "vascular structures",
                "veins and arteries",
                "microvessels"
            ],
            "tumor_skin_epithelial_bcc": [
                "Basal cell carcinoma",
                "basal cancer",
                "skin basal tumor",
                "epidermal carcinoma"
            ],
            "tumor_skin_epithelial_sqcc": [
                "Squamous cell carcinoma",
                "squamous cancer",
                "keratinocyte carcinoma",
                "skin squamous tumor"
            ],
            "tumor_skin_melanoma_melanoma": [
                "Melanoma",
                "malignant melanoma"
            ],
            "tumor_skin_naevus_naevus": [
                "Tumor-associated naevus"
            ]
    },
    'SIPAKMED':{
            "Cervix_Dyskeratotic": [
                "dyskeratotic cells",
                "premature keratinization"
            ],
            "Cervix_Koilocytotic": [
                "koilocytes",
                "koilocytotic change",
                "hpv-effected epithelium"
            ],
            "Cervix_Metaplastic": [
                "metaplasia",
                "immature metaplastic cells",
                "metaplastic squamous epithelium"
            ],
            "Cervix_Parabasal": [
                "parabasal cells",
                "immature squamous cells",
                "deep layer squamous cells"
                "small round basal cells"
            ],
            "Cervix_SuperficialIntermediate": [
                "superficial and intermediate squamous cells",
                "surface squamous epithelium",
                "surface squamous cells",
                "mature epithelial layer"
            ]
    },
    'ETI':{
            "EA": [
                "endometrial adenocarcinoma",
                "endometrial cancer",
                "malignant endometrial tumor",
                "gland-forming carcinoma"
            ],
            "EH": [
                "endometrial hyperplasia",
                "glandular overgrowth",
                "hyperplastic endometrium"
            ],
            "EP": [
                "endometrial polyp",
                "polypoid endometrium",
                "mucosal polyp"
            ],
            "Follicular": [
                "normal follicular phase",
                "follicular endometrial pattern"
            ],
            "Luteal": [
                "normal luteal phase",
                "luteal endometrial pattern"
            ],
            "Menstrual": [
                "normal menstrual phase",
                "menstrual phase mucosa"
            ]
    },
    'UBC-OCEAN':{
        "CC": [
            "clear cell carcinoma",
        ],
        "EC": [
            "endometrioid carcinoma",
            "endometrial carcinoma",
        ],
        "HGSC": [
            "high-grade serous carcinoma",
            "aggressive serous carcinoma"
        ],
        "LGSC": [
            "low-grade serous carcinoma",
        ],
        "MC": [
            "mucinous carcinoma",
        ],
        "Stroma": [
            "stroma",
            "stromal cells",
            "stromal tissue",
            "connective tissue",
            "fibrous tissue"
        ],
        "Necrosis": [
            "necrosis",
            "dead tissue",
            "cell death",
            "necrotic tissue",
        ]
    },
    'UHB':{
        'norm': [
            "benign tissue",
            "normal tissue",
            "non-cancerous tissue",
            "non-malignant tissue",],
        'tu': [
            "tumor",
            "viable tumor",
            "cancer",
        ]
    }
}