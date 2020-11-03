cat detId_to_seqId.txt  | tr ',' ' ' | grep ' 14 ' | awk '$4 == "11" {print $0}' | awk '$2 == "3" {print $0}' > layer3_endcap_ring11_modulelist.txt
