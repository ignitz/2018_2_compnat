@echo off
if [%1]==[] (
    if exist main.pdf (
        pdflatex -shell-escape main.tex
    ) else (
        pdflatex -shell-escape main.tex
        pdflatex -shell-escape main.tex
        makeindex main
        bibtex main
        pdflatex -shell-escape main.tex
        pdflatex -shell-escape main.tex
    )
) else (
    if "%1" == "clean" (
        del *.brf
        del *.blg 
        del *.out 
        del *.bbl 
        del *.log
        del *.ind
        del *.ilg
        del *.lot
        del *.lof
        del *.ind
        del *.idx
        del *.aux
        del *.toc
        del *.pdf
    )
)

