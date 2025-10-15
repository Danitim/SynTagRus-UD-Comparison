for f in datasets/{ud,str,ud_new,str_new,ud_old,str_old}/{train,dev,test}.conllu; do
  sed -i 's/\r$//' "$f"
  perl -0777 -i -pe 's/\s*\z/\n\n/s' "$f"
done