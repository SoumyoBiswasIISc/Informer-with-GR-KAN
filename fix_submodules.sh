#!/usr/bin/env bash
# 1. Remove any cached (submodule) entries recursively:
git rm -r --cached Informer2020 rational_kat_cu

# 2. Delete the nested .git folders so they become plain dirs:
rm -rf Informer2020/.git rational_kat_cu/.git

# 3. Stage all their contents as regular files:
git add Informer2020 rational_kat_cu

# 4. Commit & push:\
git commit -m "Convert Informer2020/ & rational_kat_cu/ from submodules into normal folders" && \
git push origin main && \

EOF && \
chmod +x fix_submodules.sh && \
./fix_submodules.sh




