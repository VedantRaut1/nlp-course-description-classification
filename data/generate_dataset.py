"""
Comprehensive Dataset Generator for University Course Description Classification
Expands to 10 Academic Disciplines:
1. Computer Science
2. Electronics
3. Mechanical
4. Civil
5. Business
6. Mathematics
7. Chemical Engineering
8. Biotechnology
9. Physics
10. Humanities

Dataset Split Protocol:
- Initial Split: 80% (Train + Val Pool) : 20% (Held-out Test)
- Internal Split on 80% Pool: 75% Training : 25% Validation
- Effective Final Ratios: Exactly 60% Train : 20% Validation : 20% Test
"""

import os
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DISCIPLINES = [
    "Computer Science",
    "Electronics",
    "Mechanical",
    "Civil",
    "Business",
    "Mathematics",
    "Chemical Engineering",
    "Biotechnology",
    "Physics",
    "Humanities"
]

SOURCES = [
    "MIT OpenCourseWare",
    "Coursera",
    "edX",
    "Stanford Engineering Catalog",
    "UC Berkeley Academic Guide",
    "Harvard University Course Catalog"
]

LEVELS = ["Undergraduate", "Advanced Undergraduate", "Graduate"]

# Rich domain-specific taxonomy for all 10 disciplines
CURRICULUM_DATA = {
    "Computer Science": [
        ("Design and Analysis of Algorithms", "Fundamental algorithmic design paradigms including divide-and-conquer, greedy optimization, dynamic programming, network flow algorithms, and graph traversals. Worst-case amortized time analysis and NP-completeness reductions are rigorously studied."),
        ("Advanced Operating Systems Design", "Principles of operating system architecture, process scheduling, multithreading primitives, virtual memory management, inter-process communication, file system internals, and kernel concurrency synchronization locks."),
        ("Deep Learning and Neural Representation", "Theoretical foundations and architectures of deep neural networks. Convolutional neural networks, recurrent architectures, transformer self-attention, loss surface optimization, backpropagation calculus, and GPU model training."),
        ("Relational Database Management Systems", "Database management internals, relational data algebra, declarative SQL optimization, B+ tree indexing structures, query execution plans, ACID transaction concurrency, and write-ahead logging protocols."),
        ("Computer Networks and Internet Protocols", "Architecture and layered protocols of internet systems. TCP flow and congestion control, BGP routing protocols, packet switching, DNS resolution, transport layer security (TLS), and software-defined networking."),
        ("Modern Compiler Construction", "Theory and design of programming language compilers. Lexical analysis, context-free grammars, LL/LR parsing, intermediate representations (SSA form), data-flow optimization, and machine code generation."),
        ("Principles of Cybersecurity and Cryptography", "Foundations of computer systems security. Cryptographic primitives, symmetric and asymmetric ciphers, public-key infrastructure, memory exploit defenses, buffer overflows, and zero-trust security architectures."),
        ("Distributed Cloud Systems Architecture", "Architectural principles of fault-tolerant distributed systems. Consensus algorithms (Paxos, Raft), replication consistency models, remote procedure calls (RPC), distributed key-value stores, and cloud microservices.")
    ],
    "Electronics": [
        ("Analog Integrated Circuit Design", "Analysis and design of CMOS analog integrated circuits. MOSFET small-signal modeling, single-stage and differential amplifiers, active current mirrors, frequency response, feedback stability margins, and SPICE op-amp simulation."),
        ("Digital VLSI Architecture and Design", "CMOS digital logic design and physical layout. Static and dynamic power consumption, propagation delay calculation, static timing analysis (STA), clock tree synthesis, and Verilog hardware description language."),
        ("Microprocessor Hardware and Interfacing", "Architecture of 32-bit RISC microcontrollers. Instruction pipelines, memory hierarchies, cache organization, bus protocols, interrupt controllers, timer peripherals, SPI/I2C interfaces, and low-level firmware integration."),
        ("Semiconductor Device Physics", "Quantum mechanical principles of semiconductor transport. Energy band structures, carrier generation and recombination, p-n junction electrostatics, Schottky diodes, and subthreshold MOSFET characteristics."),
        ("RF and Microwave Engineering", "High-frequency electromagnetic wave propagation in transmission lines and microstrips. Smith chart impedance matching, S-parameters, microwave filter design, high-frequency power amplifiers, and directional couplers."),
        ("Digital Signal Processing Systems", "Theory and implementation of discrete-time signals and systems. Z-transforms, discrete Fourier transform (DFT), fast Fourier transform (FFT), FIR and IIR digital filter design, and hardware DSP architectures."),
        ("Power Electronics and Energy Conversion", "Analysis of solid-state power electronic converters. Switch-mode DC-DC buck, boost, and flyback converters, inverter topologies, pulse-width modulation (PWM) control, and thermal dissipation management."),
        ("Wireless Communications and RF Systems", "Physical principles of wireless transmission. Digital modulation formats (QAM, PSK), multipath fading channel modeling, orthogonal frequency division multiplexing (OFDM), and MIMO multi-antenna beamforming.")
    ],
    "Mechanical": [
        ("Classical Applied Thermodynamics", "Fundamental thermodynamic principles for engineering systems. First and second laws, closed and control volume energy balance, entropy generation, exergy analysis, vapor power Rankine cycles, and gas turbine Brayton cycles."),
        ("Viscous Fluid Dynamics", "Continuity, momentum, and energy conservation equations in fluid motion (Navier-Stokes equations). Incompressible viscous laminar flows, boundary layer separation, turbulent pipe flow, Moody friction diagrams, and aerodynamic drag."),
        ("Conduction and Convective Heat Transfer", "Heat transfer mechanisms in mechanical systems. Steady and transient conduction, Fourier's law, forced and natural convection boundary layers, dimensionless Nusselt numbers, and heat exchanger effectiveness (NTU method)."),
        ("Mechanics of Deformable Solids", "Stress and strain tensors in structural elements. Axial loading, torsion of circular shafts, bending and shear stress distributions in beams, Mohr's circle analysis, column Euler buckling, and failure yield criteria."),
        ("Kinematics and Dynamics of Mechanisms", "Planar mechanism motion analysis. Four-bar linkages, velocity and acceleration polygon methods, dynamic force balancing, flywheel inertia determination, and balancing of rotating machinery."),
        ("Mechanical Elements Machine Design", "Design principles for mechanical machine components. Fatigue loading life calculation, hydrodynamic journal bearings, spur and helical gear rating, bolted joint preload, compression springs, clutches, and brakes."),
        ("Mechanical Vibrations and Modal Analysis", "Vibration analysis of single and multi-degree-of-freedom mechanical systems. Harmonic excitation, resonance response, vibration isolation dampers, continuous beam vibration, and modal extraction."),
        ("Manufacturing Processes and Machining", "Engineering materials processing. Metal casting, bulk deformation, sheet metal forming, chip formation mechanics in turning and milling, tool wear mechanisms, and CNC machining operations.")
    ],
    "Civil": [
        ("Structural Analysis and Matrix Methods", "Static analysis of indeterminate structural frames, trusses, and continuous beams. Moment distribution, slope deflection methods, influence lines, virtual work energy principles, and stiffness matrix formulations."),
        ("Design of Reinforced Concrete Structures", "Behavior and limit state design of reinforced concrete members. Ultimate flexural strength of beams, shear reinforcement detailing, one-way and two-way floor slabs, and axial-flexural interaction of columns."),
        ("Design of Structural Steel Buildings", "Design of steel structures according to AISC specifications. Tension members, compact and non-compact compression columns, laterally unsupported beams, bolted moment connections, and braced frames."),
        ("Geotechnical Soil Mechanics", "Fundamental physical and engineering properties of soils. Soil classification, Darcy's permeability law, 1D Terzaghi consolidation theory, effective stress, and Mohr-Coulomb shear strength criteria."),
        ("Foundation Engineering and Deep Foundations", "Geotechnical design of substructures. Ultimate bearing capacity of shallow spread footings, settlement calculations, driven pile foundations, drilled shafts, and retaining wall lateral earth pressures."),
        ("Highway Geometric Design and Planning", "Principles of highway engineering. Horizontal and vertical alignment curves, sight distance requirements, highway cross-sections, intersection channelization, and AASHTO geometric standards."),
        ("Environmental Water Treatment Engineering", "Physical, chemical, and biological unit operations for municipal water purification. Coagulation, flocculation, rapid sand filtration, chlorination disinfection, and activated sludge wastewater treatment."),
        ("Hydraulic Engineering and Open Channel Flow", "Fluid flow in open channels and pipe networks. Manning's uniform flow equation, specific energy, hydraulic jumps, gradually varied flow profile computations, and culvert hydraulic sizing.")
    ],
    "Business": [
        ("Corporate Finance and Capital Structure", "Financial decision making in corporations. Capital budgeting discounted cash flows (NPV, IRR), weighted average cost of capital (WACC), debt vs equity financing, dividend policy, and corporate governance."),
        ("Financial Statement Analysis and Accounting", "Interpretation and construction of financial statements under GAAP and IFRS. Balance sheets, income statements, cash flow statements, revenue recognition rules, inventory costing, and ratio analysis."),
        ("Strategic Marketing and Brand Positioning", "Core marketing management frameworks. Market segmentation, targeting, and brand positioning (STP), customer lifetime value, pricing strategies, promotional channels, and brand equity development."),
        ("Operations and Global Supply Chain Management", "Design and control of operational supply chains. Process bottleneck analysis, Little's Law, aggregate production planning, economic order quantity (EOQ), safety stock, and lean manufacturing."),
        ("Competitive Strategy and Industry Analysis", "Frameworks for establishing sustainable competitive advantage. Porter's Five Forces, resource-based view of the firm, value chain configuration, corporate diversification, and industry disruption."),
        ("Organizational Behavior and Leadership", "Behavioral dynamics within modern corporations. Team leadership, employee motivation theories, transformational leadership styles, organizational culture, conflict management, and change execution."),
        ("Investment Analysis and Portfolio Management", "Modern portfolio theory and capital asset pricing model (CAPM). Efficient market hypothesis, equity valuation models, fixed-income duration and convexity, hedge fund strategies, and risk metrics."),
        ("Entrepreneurial Finance and Venture Capital", "Financing emerging high-growth ventures. Term sheets, convertible notes, venture valuation methods, capitalization tables, angel financing, and startup exit strategies.")
    ],
    "Mathematics": [
        ("Real Analysis and Measure Theory", "Rigorous mathematical analysis of Euclidean spaces. Metric topologies, compactness, sequences, Cauchy convergence, Riemann-Stieltjes integration, Lebesgue measure, and dominated convergence theorem."),
        ("Abstract Algebra: Groups, Rings and Fields", "Axiomatic algebraic structures. Group homomorphisms, normal subgroups, quotient groups, Lagrange's theorem, ring ideals, unique factorization domains (UFD), and field extensions."),
        ("Complex Analysis and Conformal Mapping", "Theory of functions of a complex variable. Cauchy-Riemann equations, Cauchy's integral theorem, Taylor and Laurent series expansions, residue calculus, contour integrals, and conformal mappings."),
        ("Ordinary Differential Equations and Dynamical Systems", "Linear and nonlinear ordinary differential equations. Existence and uniqueness theorems (Picard-Lindelof), phase portrait analysis, Lyapunov stability theory, and bifurcation behavior."),
        ("Partial Differential Equations in Mathematical Physics", "Boundary value problems for linear partial differential equations. Heat equation, wave equation, Laplace equation, separation of variables, Fourier transform methods, and Green's functions."),
        ("Point-Set and Algebraic Topology", "Topological spaces and continuous mappings. Homeomorphisms, separation axioms, product topologies, connectedness, compactness, fundamental groups, and covering space theory."),
        ("Advanced Linear Algebra and Vector Spaces", "Vector spaces over arbitrary fields. Linear transformations, dual spaces, spectral theorem for normal operators, Jordan canonical forms, minimal polynomials, and inner product geometry."),
        ("Probability Theory and Mathematical Statistics", "Measure-theoretic probability foundations. Random variables, characteristic functions, laws of large numbers, central limit theorem, and maximum likelihood statistical estimation.")
    ],
    "Chemical Engineering": [
        ("Chemical Reaction Engineering and Reactor Kinetics", "Kinetics of homogeneous and heterogeneous chemical reactions. Ideal batch, continuous stirred-tank (CSTR), and plug flow reactors (PFR). Catalytic kinetics, non-isothermal reactor operation, and thermal runaway prevention."),
        ("Transport Phenomena in Chemical Systems", "Unified treatment of momentum, heat, and mass transport. Shell balances, Navier-Stokes equations, thermal conduction/convection, Fick's law of molecular diffusion, and mass transfer coefficients in boundary layers."),
        ("Chemical Engineering Thermodynamics and Phase Equilibria", "Thermodynamics of multicomponent fluid mixtures. Equations of state (Peng-Robinson), chemical potentials, fugacity coefficients, activity coefficient models (NRTL, UNIQUAC), and vapor-liquid equilibria (VLE)."),
        ("Mass Transfer Operations and Separation Processes", "Equilibrium stage and rate-based separation processes. Multicomponent distillation column design, liquid-liquid extraction, gas absorption columns, membrane separations, and McCabe-Thiele stage calculations."),
        ("Chemical Process Dynamics and Automated Control", "Mathematical modeling of dynamic chemical processes. Laplace transforms, transfer functions, feedback PID controller tuning, feedback-feedforward control, frequency response analysis, and process stability limits."),
        ("Polymer Reaction Engineering and Processing", "Polymerization kinetics and processing techniques. Step-growth and chain-growth polymerization, molecular weight distribution modeling, rheology of non-Newtonian polymer melts, and extrusion molding."),
        ("Process Safety and Plant Design in Chemical Industry", "Design and hazard mitigation in chemical production facilities. Process flow diagrams (PFD), piping and instrumentation diagrams (P&ID), HAZOP safety reviews, pressure relief sizing, and capital cost estimation."),
        ("Biochemical Reaction and Bioseparation Engineering", "Engineering analysis of biological reaction systems. Enzyme kinetics (Michaelis-Menten), bioreactor design and oxygen transfer aeration, sterilization protocols, and downstream protein purification chromatography.")
    ],
    "Biotechnology": [
        ("Molecular Biology and Recombinant DNA Technology", "Molecular mechanisms of gene expression and genetic engineering. DNA replication, transcription, translation, restriction endonucleases, plasmid cloning vectors, CRISPR-Cas9 gene editing, and PCR amplification."),
        ("Bioprocess Engineering and Microbial Fermentation", "Engineering principles of microbial cell cultivation. Stoichiometry of cell growth, batch and fed-batch fermentation kinetics, oxygen mass transfer rate (kLa), bioreactor scale-up, and continuous bioprocessing."),
        ("Cellular Biochemistry and Metabolic Pathways", "Structure and function of cellular macromolecules. Enzymatic biocatalysis, metabolic pathways (glycolysis, citric acid cycle, oxidative phosphorylation), bioenergetics, and allosteric enzyme regulation."),
        ("Bioinformatics and Computational Genomics", "Computational methods for analyzing biological sequence data. Pairwise and multiple sequence alignment (BLAST), hidden Markov models, genome assembly, phylogenetic trees, and protein structural modeling."),
        ("Immunology and Monoclonal Antibody Technology", "Principles of humoral and cell-mediated immunity. Antigen recognition, antibody structure, hybridoma technology, recombinant therapeutic monoclonal antibodies, ELISA assays, and immunotherapy design."),
        ("Stem Cell Biology and Tissue Engineering", "Mechanisms of cellular pluripotency and regenerative medicine. Embryonic and induced pluripotent stem cells, biomaterial scaffolds, cell-matrix interactions, and bioreactor culture of functional engineered tissues."),
        ("Industrial Microbiology and Downstream Processing", "Production of pharmaceuticals and industrial metabolites using microbes. Secondary metabolite biosynthesis, cell disruption techniques, centrifugation, ultrafiltration, and preparative chromatography."),
        ("Agricultural Biotechnology and Crop Genetic Engineering", "Application of genetic modification in plant science. Agrobacterium-mediated plant transformation, transgenic crop development, herbicide tolerance traits, pest resistance mechanisms, and biosafety protocols.")
    ],
    "Physics": [
        ("Classical Mechanics and Lagrangian Dynamics", "Advanced formulation of classical mechanics. Variational principles, Euler-Lagrange equations of motion, generalized coordinates, conservation laws, central force planetary motion, and Hamiltonian mechanics."),
        ("Quantum Mechanics and Atomic Structure", "Fundamental postulates of quantum mechanics. Wave-particle duality, Schrodinger wave equation, square wells, quantum harmonic oscillator, angular momentum operators, and hydrogen atom quantum states."),
        ("Electrodynamics and Classical Field Theory", "Maxwell's equations and relativistic electromagnetic fields. Gauge transformations, electromagnetic radiation from accelerating charges, wave guides, Poynting vector energy flow, and dipole radiation."),
        ("Statistical Mechanics and Thermal Physics", "Microscopic foundations of macroscopic thermodynamic behavior. Microcanonical, canonical, and grand canonical ensembles, Maxwell-Boltzmann statistics, Bose-Einstein condensation, and Fermi-Dirac electron gas."),
        ("Solid State and Condensed Matter Physics", "Electronic and crystal structure of solids. Bravais crystal lattices, X-ray Bragg diffraction, phonon vibrations, Bloch's theorem, energy band dispersion, and semiconductor electrical conduction."),
        ("Nuclear and High-Energy Particle Physics", "Structure and interactions of atomic nuclei and subatomic particles. Radioactive decay kinetics, nuclear fission and fusion reactors, quarks, leptons, gauge bosons, and Standard Model symmetries."),
        ("General Relativity and Gravitational Physics", "Geometric theory of gravitation. Tensor calculus on Riemannian manifolds, Einstein field equations, geodesic motion in curved spacetime, Schwarzschild black holes, and gravitational waves."),
        ("Laser Physics and Modern Quantum Optics", "Physics of coherent optical radiation. Spontaneous and stimulated emission, population inversion, optical resonators, laser cavity modes, photon quantization, and nonlinear optical effects.")
    ],
    "Humanities": [
        ("World Literature and Comparative Literary Theory", "Critical analysis of global literary masterpieces across historical epochs. Textual hermeneutics, narrative theory, post-colonial literary critique, modernism, and thematic representations of human identity."),
        ("Modern Political Philosophy and Ethics", "Foundational theories of justice, rights, and political governance. Classical and modern political thought including Plato, Aristotle, Hobbes, Locke, Rousseau, Kantian deontology, and utilitarianism."),
        ("Global Economic and Civilizational History", "Historical evolution of human societies and global trade networks. Rise and fall of ancient empires, the Enlightenment, Industrial Revolution, imperialism, and social transformations in the modern world."),
        ("Cognitive Linguistics and Semantic Structures", "Scientific study of human language and mental grammars. Phonology, morphological derivation, Chomskyan transformational syntax, cognitive semantics, pragmatic context, and sociolinguistics."),
        ("Cultural Anthropology and Ethnographic Inquiry", "Comparative study of human cultures, kinship systems, and social institutions. Ethnographic fieldwork methodologies, cultural relativism, ritual symbolic systems, and globalization impacts."),
        ("Epistemology and Philosophy of Mind", "Philosophical inquiry into the nature of knowledge, truth, and mental consciousness. Rationalism vs empiricism, skeptical arguments, dualism, functionalism, intentionality, and cognitive experience."),
        ("Critical Media Studies and Cultural Communication", "Critical analysis of mass media institutions, digital communication networks, and popular culture. Semiotic media decoding, ideological framing, digital disinformation, and algorithmic culture."),
        ("Art History and Visual Cultural Theory", "Evolution of visual art forms and aesthetic theories. Renaissance perspective, Baroque movement, Modernist avant-garde movements, iconography analysis, and museum curatorial practices.")
    ]
}

# Cross-disciplinary borderline courses to ensure non-trivial decision boundaries
INTERDISCIPLINARY_DATA = [
    ("Computer Science", "Computational Biology and Genome Informatics", "Algorithm development for biological big data. Sequence alignment algorithms, phylogenetic tree clustering, genome assembly heuristics, and neural network models for protein folding."),
    ("Chemical Engineering", "Biochemical Processing and Enzyme Reactors", "Application of reaction engineering principles to biological catalysts. Michaelis-Menten kinetics in CSTRs, aeration mass transfer in fermenters, and downstream recovery of bio-products."),
    ("Biotechnology", "Bioinformatics Pipeline Engineering and Transcriptomics", "High-throughput computational pipeline design for next-generation RNA sequencing data. Statistical differential expression, variant calling algorithms, and cloud genomic workflows."),
    ("Physics", "Quantum Information Theory and Quantum Gates", "Physics of quantum computing architectures. Qubits, superposition states, quantum entanglement, unitary quantum logic gates, decoherence models, and quantum error-correcting codes."),
    ("Mathematics", "Mathematical Foundations of Quantum Physics", "Rigorous functional analysis applied to quantum mechanics. Hilbert spaces, unbounded self-adjoint linear operators, spectral theorem, and mathematical formulation of observables."),
    ("Mechanical", "Micro-Electro-Mechanical Systems (MEMS) Actuators", "Multiphysics design of silicon micro-actuators. Electrostatic comb drives, piezoelectric micro-beams, cleanroom photolithography fabrication, and coupled thermal-fluid simulation."),
    ("Civil", "Environmental Fluid Mechanics and Contaminant Transport", "Fluid dynamics and mass transport in aquatic environments. Advection-diffusion equation, turbulent mixing in natural rivers, groundwater pollutant migration, and hydrologic basin modeling."),
    ("Business", "Business Ethics, Corporate Responsibility and Human Values", "Ethical decision-making frameworks for corporate executives. Moral philosophy applied to capitalism, stakeholder rights, environmental sustainability (ESG), and cross-cultural business ethics."),
    ("Humanities", "Philosophy of Artificial Intelligence and Mind", "Philosophical implications of computational machine intelligence. The Turing Test, Searle's Chinese Room argument, machine consciousness, ethics of autonomous algorithms, and the nature of human cognition.")
]

PREREQ_TEMPLATES = [
    "Prerequisites: Solid foundational coursework in the discipline and consent of the department.",
    "Prerequisites: Completion of introductory undergraduate coursework in relevant engineering or scientific disciplines.",
    "Prerequisites: Upper-division undergraduate standing and permission of the instructor.",
    "Recommended background: Prior exposure to core analytical concepts and methodological reasoning.",
    "Targeted towards undergraduate and beginning graduate students in relevant university programs."
]

def generate_dataset(samples_per_category=300):
    records = []
    print(f"Generating balanced dataset ({samples_per_category} samples per discipline across {len(DISCIPLINES)} disciplines)...")
    
    for disc in DISCIPLINES:
        core_list = CURRICULUM_DATA[disc]
        inter_list = [item for item in INTERDISCIPLINARY_DATA if item[0] == disc]
        
        for i in range(samples_per_category):
            # 88% core curricula, 12% interdisciplinary realistic cases
            if inter_list and (i % 8 == 0):
                _, title, desc = random.choice(inter_list)
            else:
                title, desc = random.choice(core_list)
                
            prefix = disc[:4].upper().replace(" ", "")
            course_num = random.randint(101, 899)
            course_id = f"{prefix}-{course_num}"
            source = random.choice(SOURCES)
            level = random.choice(LEVELS)
            prereq = random.choice(PREREQ_TEMPLATES)
            
            full_desc = f"{title}. {desc} {prereq}"
            
            records.append({
                "course_id": course_id,
                "course_title": title,
                "description": full_desc,
                "category": disc,
                "source": source,
                "level": level
            })
            
    random.shuffle(records)
    df = pd.DataFrame(records)
    
    # -------------------------------------------------------------
    # EXACT USER SPLIT REQUIREMENT:
    # 1. Initial 80 : 20 split (Train/Val pool : Held-out Test)
    # 2. Internal split on 80% pool: take 25% for Validation (25% of 80% = 20% of total)
    # 3. Final dataset distribution: 60% Train : 20% Val : 20% Test
    # -------------------------------------------------------------
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["category"],
        random_state=RANDOM_SEED
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25, # 0.25 * 0.80 = 0.20 of total
        stratify=train_val_df["category"],
        random_state=RANDOM_SEED
    )
    
    proc_dir = r"d:\nlp project\data\processed"
    raw_dir = r"d:\nlp project\data\raw"
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    df.to_csv(os.path.join(raw_dir, "university_courses_full.csv"), index=False)
    train_df.to_csv(os.path.join(proc_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(proc_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(proc_dir, "test.csv"), index=False)
    
    summary = {
        "total_courses": len(df),
        "split_ratio_description": "60% Train, 20% Validation (internal from 80% pool), 20% Held-out Test",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_percentage": round(len(train_df) / len(df) * 100, 1),
        "val_percentage": round(len(val_df) / len(df) * 100, 1),
        "test_percentage": round(len(test_df) / len(df) * 100, 1),
        "num_disciplines": len(DISCIPLINES),
        "disciplines": DISCIPLINES,
        "category_distribution_train": train_df["category"].value_counts().to_dict(),
        "category_distribution_val": val_df["category"].value_counts().to_dict(),
        "category_distribution_test": test_df["category"].value_counts().to_dict(),
        "sources": df["source"].value_counts().to_dict()
    }
    
    with open(os.path.join(proc_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[+] Successfully generated 10-Discipline Dataset:")
    print(f"    - Total Courses:      {len(df)}")
    print(f"    - Training Set (60%): {len(train_df)} courses ({len(train_df)//len(DISCIPLINES)} per discipline)")
    print(f"    - Validation Set(20%):{len(val_df)} courses ({len(val_df)//len(DISCIPLINES)} per discipline)")
    print(f"    - Testing Set (20%):  {len(test_df)} courses ({len(test_df)//len(DISCIPLINES)} per discipline)")
    print(f"    - Disciplines (10):   {', '.join(DISCIPLINES)}")

if __name__ == "__main__":
    generate_dataset(samples_per_category=300)
