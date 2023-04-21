#!/bin/bash

set -euxo pipefail

mon () {
    date +%Y-%m-%dT%H:%M:%S
    top -b -n 1 -w 512 -c -u ${2} | awk "NR==2 || / ${1}$/"
    echo
}

export -f mon

monitor () {
    setsid "$@" &
    PID=$!

    MONITOR_INTERVAL=1

    # monitor command and other processes in its process group
    TERM=linux watch --interval $MONITOR_INTERVAL --precise -x bash -c "mon $PID appuser >> process_metrics.log" >/dev/null 2>&1 &
    WATCH_PID=$!

    # monitor system resources using sysstat's sar command
    sar -o system_metrics.sar -ru $MONITOR_INTERVAL >/dev/null 2>&1 &
    SYSSTAT_PID=$!

    # monitor gpu usage
    if command -v nvidia-smi &> /dev/null
       then
	   nvidia-smi pmon -d 1 -s u -o DT -f gpu_metrics.log >/dev/null 2>&1 &
	   NVIDIASMI_PID=$!
    fi

    wait $PID
    kill -s TERM $WATCH_PID $SYSSTAT_PID || true

    if [[ -v NVIDIASMI_PID ]]
       then
	   kill -s TERM $NVIDIASMI_PID || true
    fi
}

main () {
    submission_type=$1

    if [ $submission_type = centralized ]; then
	expected_filename=solution_centralized.py
    elif [ $submission_type = federated ]; then
	expected_filename=solution_federated.py
    else
	echo "Must provide a single argument with value centralized of federated."
	exit 1
    fi

    cd /code_execution

    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -q ${expected_filename}<<<$submission_files; then
	echo "Submission zip archive must include $expected_filename"
	return 1
    fi

    echo Installed packages
    echo "######################################"
    conda list -n condaenv
    echo "######################################"

    echo Unpacking submission
    unzip ./submission/submission.zip -d ./src

    tree ./src

    if [[ $submission_type = centralized ]]; then
        echo "================ START CENTRALIZED TRAIN ================"
        monitor conda run --no-capture-output -n condaenv python main_centralized_train.py
        echo "================ END CENTRALIZED TRAIN ================"
        echo "================ START CENTRALIZED TEST ================"
        monitor conda run --no-capture-output -n condaenv python main_centralized_test.py
        echo "================ END CENTRALIZED TEST ================"

    elif [ $submission_type = federated ]; then

        while read scenario; do
            echo "================ START FEDERATED TRAIN FOR $scenario ================"
            monitor conda run --no-capture-output -n condaenv python main_federated_train.py /code_execution/data/$scenario/train/partitions.json
            echo "================ END FEDERATED TRAIN FOR $scenario ================"
            echo "================ START FEDERATED TEST FOR $scenario ================"
            monitor conda run --no-capture-output -n condaenv python main_federated_test.py /code_execution/data/$scenario/test/partitions.json
            echo "================ END FEDERATED TEST FOR $scenario ================"
        done </code_execution/data/scenarios.txt

    fi

    sadf system_metrics.sar -d -- -u | gzip > cpu_metrics.csv.gz
    sadf system_metrics.sar -d -- -r | gzip > memory_metrics.csv.gz
    gzip system_metrics.sar
    mv cpu_metrics.csv.gz memory_metrics.csv.gz system_metrics.sar.gz submission

    if [[ -f gpu_metrics.log ]]
    then
	gzip gpu_metrics.log
	mv gpu_metrics.log.gz submission
    fi

    echo "================ END ================"
}

main $1 |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

cp /code_execution/submission/log.txt /tmp/log

exit $exit_code
