## INSTALL
``` bash
mkdir -p ~/.vim/{autoload,bundle,plugin,colors,undoinfo}
curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
wget http://www.vim.org/scripts/download_script.php?src_id=13834 -O ~/.vim/plugin/toggle.vim

cd ~/.vim/colors
wget -O wombat256mod.vim http://www.vim.org/scripts/download_script.php?src_id=13400

rm -f ~/.bashrc && ln -s ~/syncfiles/dotfiles/bashrc ~/.bashrc
rm -f ~/.screenrc && ln -s ~/syncfiles/dotfiles/screenrc ~/.screenrc
rm -f ~/.vimrc && ln -s ~/syncfiles/dotfiles/vimrc ~/.vimrc
rm -f ~/.inputrc && ln -s ~/syncfiles/dotfiles/inputrc ~/.inputrc
# now get italics via https://alexpearce.me/2014/05/italics-in-iterm2-vim-tmux/
# and install plugins in vim via :PlugInstall
# and get monaco patched font for Iterm from https://github.com/mattamizer/patched-monaco/blob/master/Monaco%20for%20Powerline.otf
```

## PYFILES
### miscutils.py
Contains various functions that would find common use in python. My hack for allowing importing of miscutils would be:
``` python
import sys, os
sys.path.append(os.getenv("HOME") + '~/syncfiles/pyfiles')
```
or we can modify the pythonpath variable for this (included in bashrc)

### tabletex.py
Suppose we had a text file (test.txt) with the contents (note that the spacing doesn't have to look like this)
```
col1 | col2          | col3 | col4

1    | 2             | 3    | 4
4    | multirow 3 10 | 8    | multirow 2 $\met$
7    | -             | -    | -
7    | -             | -    | -
1    | 2             | 3    | -
```
and we wanted to make a nice LaTeX table from it. Well, now you can. Simply do `cat test.txt | python tabletex.py` to get
the TeXified source. To go a step further, you could do `cat test.txt | python tabletex.py | pdflatex; pdfcrop texput.pdf output.pdf`.
The syntax is as follows:
- columns are separated by |
- - indicates an empty entry
- a blank line will cause the script to draw two horizontal lines instead of one
- "multirow [x] [y]" will join [x] rows starting with the current and put the content [y] inside

### stats.py
Takes piped input and prints out length, mean, sigma, sum, min, max. It can ignore non-numerical lines, but it only handles 1 column. If specified, the first argument of stats.py provides the column of piped input to use
``` bash
seq 1 1 10 | stats
```
produces
```
        length: 10
        mean:   5.5
        sigma:  3.0276503541
        sum:    55.0
        min:    1.0
        max:    10.0
```
Additionally, if no numbers are detected, but a few text objects are found, it will output a frequency histogram of the text (column specification also works for this).
``` bash
ls -l | stats 6
```
produces
```
Found 36 words, so histo will be made!
Apr | ********* (9)
Mar | ******** (8)
Feb | ***** (5)
Aug | **** (4)
May | **** (4)
Jun | ** (2)
Jul | ** (2)
Dec | ** (2)
```
