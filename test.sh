for t in /base_scan /base_scan_inf; do
  echo "=== $t ==="
  ros2 topic echo "$t" --once | awk '
    /^range_max:/ {rm=$2+0}
    /^ranges:/ {in=1; next}
    /^intensities:/ {in=0}
    in{
      gsub(/[\[\],]/,"")
      for(i=1;i<=NF;i++){
        if($i==""||$i=="nan") continue
        if($i=="inf"){inf++; continue}
        v=$i+0; n++
        if(!seen||v>mx){mx=v; seen=1}
        if(v>=rm-0.01)c01++
        if(v>=rm-0.02)c02++
        if(v>=rm-0.05)c05++
      }
    }
    END{
      printf("range_max=%.3f finite_max=%.3f inf=%d finite=%d >=rm-0.01=%d >=rm-0.02=%d >=rm-0.05=%d\n",
             rm,mx,inf,n,c01,c02,c05)
    }'
done
