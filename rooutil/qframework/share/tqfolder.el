;; this is an implementation of a major mode
;; for editing TQFolder txt files 
;;
;; it was written based on the following tutorial
;;    http://www.emacswiki.org/emacs/ModeTutorial
;; the following was also used for reference
;;    http://www.emacswiki.org/emacs/spss.el
;;
;; to import this file, put the following line into your .emacs file
;;    (load-file "path/to/this/file")
;; to enable autoloading this mode for some specific txt file
;; put the following as the first line in the file
;;    # -*- mode: tqfolder; -*-


;;; Code:
(defvar tqfolder-mode-hook nil)
(defvar tqfolder-mode-map
  (let ((tqfolder-mode-map (make-keymap)))
    (define-key tqfolder-mode-map "\C-j" 'newline-and-indent)
    tqfolder-mode-map)
  "Keymap for TQFOLDER major mode")

(add-to-list 'auto-mode-alist '("\\.wpd\\'" . tqfolder-mode))

(defconst tqfolder-font-lock-keywords
  (list
;   '( . font-lock-builtin-face)
   '("=\\|+\\|<\\|>\\|@\\|?\\|,\\|$delete\\|$copy\\|$move\\|$escape\\|$modify\\|$ignore\\|$import\\|$for\\|$create\\|$include\\|$replace\\|$print\\|$printline\\|$write" . font-lock-keyword-face)
   '("\\('\\w*'\\)" . font-lock-variable-name-face)
   '("\\<\\(true\\|false\\)\\>" . font-lock-constant-face)
   )
  "Minimal highlighting expressions for TQFOLDER mode.")


(defun tqfolder-indent-line ()
  "Indent current line as tqfolder code."
  (interactive)
  (beginning-of-line)
  (if (bobp) 
      ;; if at beginning of buffer, indent to start of line
      (indent-line-to 0)
    (let ((not-indented t) cur-indent)
      (if (looking-at "^[ \t]*}")
		  ;; if we are currently at the end of a block reduce indentation
		  (progn
			(save-excursion
			  (forward-line -1)
			  (setq cur-indent (- (current-indentation) default-tab-width)))
			(if (< cur-indent 0)
				(setq cur-indent 0)))
		(save-excursion
		  ;; if we are not at the end of a block search backwards through buffer
		  (while not-indented
			(forward-line -1)
			;;			(prin1 (what-line)) ;; uncomment this line for debug printout
			(if (looking-at "[\t ]*}")
				;; if we find a block ending then we are not in a block, 
				;; so indent to same level as block ending
				(progn
				  (setq cur-indent (current-indentation))
				  (setq not-indented nil))
			  (if (looking-at "[^\"#\n]*}")
				  ;; if we find a block ending then we are not in a block, 
				  ;; so indent to same level as block ending
				  (progn
					(setq cur-indent (- (current-indentation) default-tab-width))
					(setq not-indented nil))
				(if (looking-at "[^\"#\n]*{")
					;; if we find a block start then we are in a block,
					;; so indent a bit further
					(progn
					  (setq cur-indent (+ (current-indentation) default-tab-width))
					  (setq not-indented nil))
				  (if (bobp)
					  ;; if there were no blocks before current position then don't change indentation
					  (setq not-indented nil)))))))
		)
      (if cur-indent
		  (indent-line-to cur-indent)
		(indent-line-to 0)))))

(defvar tqfolder-mode-syntax-table
  (let ((tqfolder-mode-syntax-table (make-syntax-table)))
										; This is added so entity names with underscores can be more easily parsed
    (modify-syntax-entry ?_ "w" tqfolder-mode-syntax-table)
										; Comment styles are same as C++
    (modify-syntax-entry ?# "<" tqfolder-mode-syntax-table)
    (modify-syntax-entry ?\n ">" tqfolder-mode-syntax-table)
    tqfolder-mode-syntax-table)
  "Syntax table for tqfolder-mode")

(defun tqfolder-mode ()
  (interactive)
  (kill-all-local-variables)
  (use-local-map tqfolder-mode-map)
  (set-syntax-table tqfolder-mode-syntax-table)
  ;; Set up font-lock
  (set (make-local-variable 'font-lock-defaults) '(tqfolder-font-lock-keywords))
  ;; Register our indentation function
  (set (make-local-variable 'indent-line-function) 'tqfolder-indent-line)  
  (setq major-mode 'tqfolder-mode)
  (setq mode-name "TQFOLDER")
  (setq default-tab-width 2) 
  (run-hooks 'tqfolder-mode-hook))



