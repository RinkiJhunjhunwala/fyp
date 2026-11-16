import os

thesis_dir = 'thesis'

# 1. Read individual files
with open(os.path.join(thesis_dir, 'chapter1.tex'), 'r', encoding='utf-8') as f:
    ch1 = f.read()
with open(os.path.join(thesis_dir, 'chapter2.tex'), 'r', encoding='utf-8') as f:
    ch2 = f.read()
with open(os.path.join(thesis_dir, 'chapter3.tex'), 'r', encoding='utf-8') as f:
    ch3 = f.read()
with open(os.path.join(thesis_dir, 'chapter4.tex'), 'r', encoding='utf-8') as f:
    ch4 = f.read()
with open(os.path.join(thesis_dir, 'chapter5.tex'), 'r', encoding='utf-8') as f:
    ch5 = f.read()
with open(os.path.join(thesis_dir, 'chapter6.tex'), 'r', encoding='utf-8') as f:
    ch6 = f.read()
with open(os.path.join(thesis_dir, 'appendixA.tex'), 'r', encoding='utf-8') as f:
    appA = f.read()

# 2. Formatted bibliography that requires ZERO external bibtex passes
inlined_bibliography = r"""\begin{thebibliography}{99}

\bibitem{allgower1967shock}
M.~Allg{\"o}wer and C.~Burri.
\newblock {``Schockindex''}.
\newblock \emph{Deutsche Medizinische Wochenschrift}, 92(43):1947--1950, 1967.

\bibitem{antman2000timi}
E.~M. Antman, M.~Cohen, P.~J. Bernink, C.~H. McCabe, T.~Horacek, G.~Papuchis,
  B.~Mautner, R.~Corbalan, D.~Radley, and E.~Braunwald.
\newblock The {TIMI} risk score for unstable angina/non-{ST} elevation {MI}: a
  method for prognostication and selecting therapy.
\newblock \emph{JAMA}, 284(7):835--842, 2000.

\bibitem{bonawitz2017practical}
K.~Bonawitz, V.~Ivanov, B.~Kreuter, A.~Marcedone, H.~B. McMahan, S.~Patel,
  D.~Ramage, A.~Segal, and K.~Seth.
\newblock Practical secure aggregation for privacy-preserving machine learning.
\newblock In \emph{Proceedings of the 2017 ACM SIGSAC Conference on Computer
  and Communications Security}, pages 1175--1191, 2017.

\bibitem{breiman2001random}
L.~Breiman.
\newblock Random forests.
\newblock \emph{Machine Learning}, 45(1):5--32, 2001.

\bibitem{dayan2021federated}
I.~Dayan, H.~R. Roth, A.~Zhong, A.~Harouni, A.~Gentili, A.~Z. Abidin, A.~Liu,
  A.~B. Costa, B.~J. Wood, C.-S. Tsai, et~al.
\newblock Federated learning for predicting clinical outcomes in patients with
  {COVID-19}.
\newblock \emph{Nature Medicine}, 27(10):1735--1743, 2021.

\bibitem{dwork2014algorithmic}
C.~Dwork and A.~Roth.
\newblock The algorithmic foundations of differential privacy.
\newblock \emph{Foundations and Trends{\textregistered} in Theoretical
  Computer Science}, 9(3--4):211--407, 2014.

\bibitem{fox2006grace}
K.~A. Fox, O.~H. Dabbous, R.~J. Goldberg, K.~S. Pieper, K.~A. Eagle, F.~Van~de
  Werf, A.~Avezum, S.~G. Goodman, M.~D. Flather, F.~A. Anderson, et~al.
\newblock Prediction of risk of death and myocardial infarction in the six
  months after presentation with acute coronary syndrome: prospective
  multinational observational study ({GRACE}).
\newblock \emph{BMJ}, 333(7578):1091, 2006.

\bibitem{goldberger2000physiobank}
A.~L. Goldberger, L.~A. Amaral, L.~Glass, J.~M. Hausdorff, P.~C. Ivanov, R.~G.
  Mark, J.~E. Mietus, G.~B. Moody, C.-K. Peng, and H.~E. Stanley.
\newblock {PhysioBank, PhysioToolkit, and PhysioNet}: components of a new
  research resource for complex physiologic signals.
\newblock \emph{Circulation}, 101(23):e215--e220, 2000.

\bibitem{johnson2023mimic}
A.~E. Johnson, L.~Bulgarelli, L.~Shen, A.~Gayles, A.~Shammout, S.~Horng, T.~J.
  Pollard, S.~Hao, B.~Moody, B.~Gow, et~al.
\newblock {MIMIC-IV}, a freely accessible electronic health record dataset.
\newblock \emph{Scientific Data}, 10(1):1, 2023.

\bibitem{kaissis2020secure}
G.~A. Kaissis, M.~R. Makowski, D.~R{\"u}ckert, and R.~F. Braren.
\newblock Secure, privacy-preserving and federated machine learning in medical
  imaging.
\newblock \emph{Nature Machine Intelligence}, 2(6):305--311, 2020.

\bibitem{krumholz2018cardiovascular}
H.~M. Krumholz.
\newblock Cardiovascular disease and the dawn of artificial intelligence.
\newblock \emph{Circulation: Cardiovascular Quality and Outcomes}, 11(8):e005089, 2018.

\bibitem{li2020federated}
T.~Li, A.~K. Sahu, M.~Zaheer, M.~Sanjabi, A.~Talwalkar, and V.~Smith.
\newblock Federated optimization in heterogeneous networks.
\newblock \emph{Proceedings of Machine Learning and Systems}, 2:429--450, 2020.

\bibitem{mcmahan2017communication}
B.~McMahan, E.~Moore, D.~Ramage, S.~Hampson, and B.~Aguera~y Arcas.
\newblock Communication-efficient learning of deep networks from decentralized data.
\newblock In \emph{Artificial Intelligence and Statistics}, pages 1273--1282. PMLR, 2017.

\bibitem{pedregosa2011scikit}
F.~Pedregosa, G.~Varoquaux, A.~Gramfort, V.~Michel, B.~Thirion, O.~Grisel,
  M.~Blondel, P.~Prettenhofer, R.~Weiss, V.~Dubourg, et~al.
\newblock Scikit-learn: Machine learning in {Python}.
\newblock \emph{Journal of Machine Learning Research}, 12:2825--2830, 2011.

\bibitem{rajaram2022privacy}
R.~Rajaram, A.~Garg, and S.~Reddy.
\newblock Privacy-preserving federated machine learning for clinical cardiovascular event detection.
\newblock \emph{IEEE Journal of Biomedical and Health Informatics}, 26(9):4680--4689, 2022.

\bibitem{rieke2020future}
N.~Rieke, J.~Hancox, W.~Li, F.~Milletar{\`\i}, H.~R. Roth, S.~Albarqouni,
  S.~Bakas, M.~N. Galtier, B.~A. Landman, K.~Maier-Hein, et~al.
\newblock The future of digital health with federated learning.
\newblock \emph{NPJ Digital Medicine}, 3(1):1--7, 2020.

\bibitem{shokri2017membership}
R.~Shokri, M.~Stronati, C.~Song, and V.~Shmatikov.
\newblock Membership inference attacks against machine learning models.
\newblock In \emph{2017 IEEE Symposium on Security and Privacy (SP)}, pages 3--18. IEEE, 2017.

\bibitem{thygesen2018fourth}
K.~Thygesen, J.~S. Alpert, A.~S. Jaffe, B.~R. Chaitman, J.~J. Bax, D.~A. Morrow,
  and H.~D. White.
\newblock Fourth universal definition of myocardial infarction (2018).
\newblock \emph{Journal of the American College of Cardiology}, 72(18):2231--2264, 2018.

\end{thebibliography}"""

# 3. Super fast, clean preamble without slow packages like microtype
preamble = r"""\documentclass[12pt,a4paper,oneside]{report}

\usepackage[a4paper,left=1.5in,right=1.0in,top=1.0in,bottom=1.0in]{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}

% 1.5 line spacing as per NIT Agartala thesis guidelines
\onehalfspacing

% Paragraph formatting
\setlength{\parindent}{0.4in}
\setlength{\parskip}{6pt plus 2pt minus 1pt}

% Section title formatting
\titleformat{\chapter}[display]
  {\normalfont\Large\bfseries\centering}
  {\MakeUppercase{\chaptertitlename}\ \thechapter}{15pt}{\Huge\bfseries}
\titlespacing*{\chapter}{0pt}{-20pt}{25pt}

\titleformat{\section}
  {\normalfont\Large\bfseries}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}

\titleformat{\subsection}
  {\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{3pt}

% Headers and Footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\small\nouppercase{\rightmark}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% Preliminary environments
\newenvironment{dedication}
{
  \cleardoublepage
  \thispagestyle{plain}
  \begin{center}
    \vspace*{2.5in}
    {\Large\bfseries Dedication}\par
    \vspace{0.3in}
  \end{center}
  \begin{itshape}
  \centering
}
{
  \end{itshape}
  \vfill
}

\newenvironment{thesisquote}
{
  \cleardoublepage
  \thispagestyle{plain}
  \vspace*{2.8in}
  \begin{center}
  \begin{itshape}\large
}
{
  \end{itshape}
  \end{center}
  \vfill
}

\newenvironment{acknowledgements}
{
  \cleardoublepage
  \thispagestyle{plain}
  \begin{center}
    {\Large\bfseries Acknowledgement}\par
    \vspace{0.3in}
  \end{center}
}
{
  \vfill
}

\begin{document}

% ==============================================================================
% 1. COVER PAGE
% ==============================================================================
\begin{titlepage}
    \centering
    \vspace*{0.5cm}
    
    {\Huge \textbf{PRIVACY-PRESERVING CARDIOVASCULAR DISEASE CLASSIFICATION USING FEDERATED LEARNING\par}}
    
    \vspace{2.2cm}
    
    \includegraphics[width=4.2cm]{figures/nita_logo.png}
    
    \vspace{2.5cm}
    
    {\large \textit{A Final Year Project Report Submitted by}\par}
    \vspace{0.4cm}
    {\Large \textbf{Rinki Jhunjhunwala}\par}
    {\large \textbf{(Enrollment No.: 21UEC024)}\par}
    
    \vfill
    
    {\large \textbf{DEPARTMENT OF COMPUTER SCIENCE \& ENGINEERING}\par}
    \vspace{0.15cm}
    {\large \textbf{NATIONAL INSTITUTE OF TECHNOLOGY, AGARTALA}\par}
    \vspace{0.15cm}
    {\normalsize TRIPURA, INDIA -- 799046\par}
    \vspace{0.3cm}
    {\large \textbf{May, 2026}\par}
    
\end{titlepage}

% ==============================================================================
% 2. INNER TITLE PAGE
% ==============================================================================
\cleardoublepage
\thispagestyle{empty}
\begin{center}
    \vspace*{0.5cm}
    
    {\Huge \textbf{PRIVACY-PRESERVING CARDIOVASCULAR DISEASE CLASSIFICATION USING FEDERATED LEARNING\par}}
    
    \vspace{1.5cm}
    
    {\large \textit{Report submitted to}\par}
    \vspace{0.2cm}
    {\Large \textbf{National Institute of Technology, Agartala}\par}
    \vspace{0.2cm}
    {\large \textit{for the award of the degree of}\par}
    \vspace{0.3cm}
    {\Large \textbf{Bachelor of Technology}\par}
    \vspace{0.2cm}
    {\large \textit{in}\par}
    \vspace{0.2cm}
    {\Large \textbf{Computer Science \& Engineering}\par}
    
    \vspace{1.2cm}
    
    {\large \textit{by}\par}
    \vspace{0.2cm}
    {\Large \textbf{Rinki Jhunjhunwala}\par}
    {\normalsize \textbf{(Enrollment No.: 21UEC024)}\par}
    
    \vspace{1.2cm}
    
    {\large \textit{Under the Guidance of}\par}
    \vspace{0.2cm}
    {\Large \textbf{Dr. Nirmalya Kar}\par}
    {\normalsize Associate Professor\par}
    {\normalsize Department of Computer Science \& Engineering\par}
    {\normalsize National Institute of Technology, Agartala\par}
    
    \vspace{0.8cm}
    \includegraphics[width=2.5cm]{figures/nita_logo.png}
    \vspace{0.4cm}
    
    \vfill
    
    {\normalsize \textbf{DEPARTMENT OF COMPUTER SCIENCE \& ENGINEERING}\par}
    {\normalsize \textbf{NATIONAL INSTITUTE OF TECHNOLOGY, AGARTALA}\par}
    {\normalsize \textbf{May, 2026}\par}
\end{center}

% Set frontmatter page numbering
\pagenumbering{roman}
\setcounter{page}{3}

% ==============================================================================
% 3. DEDICATION
% ==============================================================================
\begin{dedication}
    This thesis is affectionately dedicated to our beloved parents, whose endless sacrifices, unwavering faith, and continuous prayers have been our greatest source of strength throughout this academic journey;\\
    \vspace{0.8cm}
    to Prof. Mrinal Kanti Debbarma, Head of Department, and our supervisor Dr. Nirmalya Kar, for their brilliant mentorship, academic wisdom, and patience;\\
    \vspace{0.8cm}
    and to all the healthcare professionals and critical-care researchers who work tirelessly to protect human life.
\end{dedication}

% ==============================================================================
% 4. EPIGRAPH / QUOTE
% ==============================================================================
\begin{thesisquote}
    \vspace*{1.5in}
    \begin{quote}
        \centering
        \Large \itshape
        ``The only way to discover the limits of the possible is to go beyond them into the impossible.''
        
        \vspace{0.5cm}
        \normalfont \large --- \textbf{Arthur C. Clarke}
    \end{quote}
\end{thesisquote}

% ==============================================================================
% 5. REPORT APPROVAL FOR B.TECH
% ==============================================================================
\cleardoublepage
\thispagestyle{plain}
\begin{center}
    \vspace*{0.5in}
    {\Large \textbf{REPORT APPROVAL FOR B.TECH}\par}
    \vspace{1.0in}
\end{center}

\noindent This report entitled \textbf{``PRIVACY-PRESERVING CARDIOVASCULAR DISEASE CLASSIFICATION USING FEDERATED LEARNING''}, submitted by \textbf{Rinki Jhunjhunwala (Enrollment No.: 21UEC024)}, is approved in partial fulfillment of the requirements for the award of \textbf{Bachelor of Technology} in \textbf{Computer Science \& Engineering}.

\vspace{3.5in}

\begin{table}[h]
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc}
    \rule{6.5cm}{0.4pt} & \rule{6.5cm}{0.4pt} \\
    \textbf{Dr. Nirmalya Kar} & \textbf{Prof. Mrinal Kanti Debbarma} \\
    (Project Supervisor) & (Head of the Department) \\
    Associate Professor & Professor \\
    Department of CSE & Department of CSE \\
    NIT Agartala & NIT Agartala
\end{tabular*}
\end{table}

\vspace{1.0cm}
\noindent \textbf{Date:} \rule{4cm}{0.4pt}\\
\noindent \textbf{Place:} NIT Agartala

% ==============================================================================
% 6. CANDIDATE'S DECLARATION
% ==============================================================================
\cleardoublepage
\thispagestyle{plain}
\begin{center}
    \vspace*{0.5in}
    {\Large \textbf{DECLARATION}\par}
    \vspace{0.8in}
\end{center}

\noindent I declare that the work presented in this project report entitled \textbf{``PRIVACY-PRESERVING CARDIOVASCULAR DISEASE CLASSIFICATION USING FEDERATED LEARNING''}, submitted to the Department of Computer Science and Engineering, National Institute of Technology, Agartala, for the award of the degree of \textbf{Bachelor of Technology in Computer Science \& Engineering}, represents my original ideas and research in my own words. Where others' ideas, data, or published findings have been included, they have been adequately cited and referenced according to accepted academic standards.

\vspace{0.4cm}
\noindent I also declare that I have adhered to all principles of academic honesty and ethical integrity and have not misrepresented, fabricated, or falsified any clinical data, experimental observation, or software code in this submission. I understand that any violation of the above will be cause for disciplinary action by the Institute and may also evoke legal consequences from sources that have not been properly acknowledged.

\vspace{2.5in}

\begin{flushright}
    \rule{6cm}{0.4pt}\\
    \textbf{Rinki Jhunjhunwala}\\
    Enrollment No.: 21UEC024\\
    Department of CSE, NIT Agartala
\end{flushright}

\vspace{0.8cm}
\noindent \textbf{Date:} May, 2026\\
\noindent \textbf{Place:} Agartala, India

% ==============================================================================
% 7. SUPERVISOR CERTIFICATE
% ==============================================================================
\cleardoublepage
\thispagestyle{plain}
\begin{center}
    \vspace*{0.5in}
    {\Large \textbf{CERTIFICATE}\par}
    \vspace{0.8in}
\end{center}

\noindent This is to certify that the work contained in the final year project report entitled \textbf{``PRIVACY-PRESERVING CARDIOVASCULAR DISEASE CLASSIFICATION USING FEDERATED LEARNING''}, submitted by \textbf{Rinki Jhunjhunwala (Enrollment No.: 21UEC024)}, has been carried out under my direct supervision and guidance in the Department of Computer Science \& Engineering, National Institute of Technology, Agartala.

\vspace{0.5cm}
\noindent To the best of my knowledge, this report represents the candidate's authentic research effort and has not been submitted elsewhere for the award of any degree, diploma, or fellowship.

\vspace{3.0in}

\begin{flushright}
    \rule{6.5cm}{0.4pt}\\
    \textbf{Dr. Nirmalya Kar}\\
    Associate Professor\\
    Department of Computer Science \& Engineering\\
    National Institute of Technology, Agartala\\
    West Tripura, PIN -- 799046, India
\end{flushright}

% ==============================================================================
% 8. ACKNOWLEDGEMENTS
% ==============================================================================
\begin{acknowledgements}
    I wish to express my profound sense of gratitude and respect to my project supervisor, \textbf{Dr. Nirmalya Kar}, Associate Professor, Department of Computer Science \& Engineering, National Institute of Technology, Agartala. His intellectual guidance, continuous encouragement, constructive criticisms, and inspirational attitude provided the foundation upon which this research was conceived and completed.

    I am deeply indebted to \textbf{Prof. Mrinal Kanti Debbarma}, Head of the Department of Computer Science \& Engineering, for his visionary leadership, support, and for providing the computing infrastructure necessary to conduct high-performance simulations.

    I also convey my sincere thanks to \textbf{Prof. (Dr.) S. K. Patra}, Director, National Institute of Technology, Agartala, for fostering an academic environment dedicated to high-quality technological inquiry and undergraduate research.

    I extend my heartfelt appreciation to all faculty members, technical officers, and staff of the Computer Science \& Engineering Department for their valuable cooperation throughout my four years of undergraduate study.

    Finally, words fail to express my eternal gratitude to my parents and family members for their boundless love, patient sacrifices, and moral support, which inspired me every day to pursue academic excellence.
    
    \vspace{1.5in}
    \begin{flushright}
        \textbf{Rinki Jhunjhunwala}\\
        (Enrollment No.: 21UEC024)
    \end{flushright}
\end{acknowledgements}

% ==============================================================================
% 9. LIST OF FIGURES & TABLES
% ==============================================================================
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{List of Figures}
\listoffigures

\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{List of Tables}
\listoftables

% ==============================================================================
% 10. ABSTRACT
% ==============================================================================
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{Abstract}
\begin{center}
    {\Large \textbf{Abstract}\par}
    \vspace{0.4in}
\end{center}

Acute Myocardial Infarction (AMI) remains among the leading global causes of mortality and critical-care admissions. Although machine learning holds tremendous promise for detecting early hemodynamic deterioration patterns from routine physiological monitors, clinical adoption is severely constrained by strict data privacy regulations, including the Health Insurance Portability and Accountability Act (HIPAA), the European General Data Protection Regulation (GDPR), and India's Digital Personal Data Protection (DPDP) Act. These frameworks prohibit the centralized aggregation of identifiable electronic health records across institutional boundaries, creating isolated data silos that limit model generalizability.

In this thesis, we propose and validate an end-to-end privacy-preserving framework for cardiovascular disease classification using Federated Learning. Using the Medical Information Mart for Intensive Care IV (MIMIC-IV v3.1) database, we extracted an initial candidate population of 90 patients across ICD-9 (code 410) and ICD-10 (codes I21--I22) cardiovascular cohorts. A rigorous data-integrity inspection revealed that 10 candidate records possessed clinical events recorded strictly outside ICU charted telemetry. Rather than imputing synthetic observations, we established a verified ICU cohort of 80 unique patients symmetrically balanced between AMI-positive and control cases. Over an explicit 24-hour observation window, we engineered 43 statistical, temporal trajectory (ordinary least squares trend slopes), and hemodynamic strain features (Shock Index, Pulse Pressure) from timestamped heart rate, systolic blood pressure, diastolic blood pressure, and oxygen saturation ($\text{SpO}_2$) series.

We first benchmarked six centralized machine learning models using leak-free 5-fold Stratified Cross-Validation. Demonstrating the critical impact of sample parsimony, a linear Logistic Regression model with $L_2$ regularization outperformed complex non-linear architectures (Random Forest 62.50\%, Gradient Boosting 56.25\%, KNN 50.00\%), achieving a test accuracy of 68.75\%, sensitivity of 75.00\%, and an Area Under the ROC Curve (ROC-AUC) of 0.7188.

Next, we simulated a distributed healthcare ecosystem across three distinct hospital nodes. We proved that training in isolation within data-scarce community hospitals leads to catastrophic sample-deprivation collapse: Hospital B ($N=16$) collapsed to a test accuracy of 31.25\% with 0.0\% sensitivity. By implementing the Federated Averaging (FedAvg) algorithm, decentralized collaborative training restored global test accuracy to 68.75\% under IID partitioning and 75.00\% under non-IID partitioning without transferring a single raw patient telemetry record outside the hospital perimeter. Finally, we deployed the federated model into an interactive clinical web platform providing real-time inference and risk attribution. This investigation demonstrates that federated learning effectively breaks the data-silo deadlock, democratizing high-performing clinical intelligence for resource-constrained hospitals while guaranteeing strict patient confidentiality.

% ==============================================================================
% 11. TABLE OF CONTENTS
% ==============================================================================
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{Contents}
\tableofcontents

% ==============================================================================
% 12. MAIN CHAPTERS (ARABIC PAGE NUMBERING)
% ==============================================================================
\cleardoublepage
\pagenumbering{arabic}
\setcounter{page}{1}
"""

postamble = r"""
% ==============================================================================
% 13. REFERENCES
% ==============================================================================
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{References}
""" + inlined_bibliography + r"""

% ==============================================================================
% 14. APPENDIX
% ==============================================================================
\appendix
""" + appA + r"""

\end{document}
"""

fast_complete_text = preamble + ch1 + "\n\n" + ch2 + "\n\n" + ch3 + "\n\n" + ch4 + "\n\n" + ch5 + "\n\n" + ch6 + "\n\n" + postamble

with open(os.path.join(thesis_dir, 'thesis_complete.tex'), 'w', encoding='utf-8') as f:
    f.write(fast_complete_text)

print(f"Generated fast thesis_complete.tex! Size: {len(fast_complete_text)} bytes")

# Also update thesis_nita.tex to remove microtype
with open(os.path.join(thesis_dir, 'thesis_nita.tex'), 'r', encoding='utf-8') as f:
    tn = f.read()
tn = tn.replace('\\usepackage{microtype}', '% \\usepackage{microtype} (disabled for fast Overleaf compilation)')
with open(os.path.join(thesis_dir, 'thesis_nita.tex'), 'w', encoding='utf-8') as f:
    f.write(tn)

print("Updated thesis_nita.tex successfully!")
