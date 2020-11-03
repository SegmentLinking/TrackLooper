for DETID in $(cat layer3_endcap_ring11_modulelist.txt | awk '{print $1}' | sed 'N;s/\n/ /' | tr ' ' ',');
# for DETID in ${DETIDs};
do
    RESULT="NONE"
    for iDETID in $(echo ${DETID} | tr ',' ' ');
    do
        RESULT="${RESULT} $(cat scripts/module_connection_map_data.txt | awk '$1 == "'$iDETID'" {print $0}')"
    done

    echo ${RESULT};
done
