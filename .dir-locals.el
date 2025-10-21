;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((haskell-mode . ((eval . (format-all-mode))
                  (eval . (with-eval-after-load 'lsp-mode
                            (add-to-list 'lsp-file-watch-ignored-directories "[/\\\\]\\testdata\\'")
                            (add-to-list 'lsp-file-watch-ignored-directories "[/\\\\]\\deps\\'"))))))
