for f in datasets/{ud,str}/{train,dev,test}.conllu; do
  sed -i 's/\r$//' "$f"
  perl -0777 -i -pe 's/\s*\z/\n\n/s' "$f"
done