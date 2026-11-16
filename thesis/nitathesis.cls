%%
%% nitathesis.cls - LaTeX Class for NIT Agartala B.Tech/M.Tech/Ph.D. Thesis
%% Based on N.I.T. Agartala Thesis Format by Anupam Jamatia (2007, updated 2012)
%% Optimized for compatibility with modern pdfLaTeX on Overleaf
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nitathesis}[2012/03/30 v0.1beta NIT Agartala Thesis Class]

\LoadClass[12pt,a4paper,oneside]{report}

\RequirePackage[a4paper,left=1.5in,right=1.0in,top=1.0in,bottom=1.0in]{geometry}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{xcolor}
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[pdftex,colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}

% 1.5 line spacing as per NIT Agartala thesis guidelines
\onehalfspacing

% Typographical tolerances to eliminate overfull/underfull hbox warnings
\emergencystretch 3em
\hfuzz 3pt
\vfuzz 3pt
\hbadness 10000


% Paragraph indentation
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
\setlength{\headheight}{14.5pt}
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

% Frontmatter definitions
\newcommand{\rollnumber}[1]{\def\@rollnumber{#1}}
\newcommand{\department}[1]{\def\@department{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\supervisortitle}[1]{\def\@supervisortitle{#1}}
\newcommand{\hod}[1]{\def\@hod{#1}}
\newcommand{\hodtitle}[1]{\def\@hodtitle{#1}}
\newcommand{\degree}[1]{\def\@degree{#1}}
\newcommand{\degreediscipline}[1]{\def\@degreediscipline{#1}}
\newcommand{\degreedate}[1]{\def\@degreedate{#1}}

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

\endinput
