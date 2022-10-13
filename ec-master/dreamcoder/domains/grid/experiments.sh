# Very verbose, basically everything that isn't a very noisy status message.
# export OCAMLRUNPARAM=v=0x5BD

SEED=334419183

# https://stackoverflow.com/questions/3004811/how-do-you-run-multiple-programs-in-parallel-from-a-bash-script
trap 'kill 0' SIGINT

function max_jobs {
   while [ `jobs -r -p | wc -l` -ge $1 ]; do
      sleep 1
   done
}

function exp() {
  # HACK: we randomly generate a log file name, to make sure they aren't overwritten
  # by other concurrent processes.
  LOGFILE=logoutput/$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32).log
  touch $LOGFILE
  command singularity-venv/bin/python3 -m dreamcoder.domains.grid.grid -c 10 --compressionCPUs 1 -i 4 --enumerationTimeout 120 "$@" --log_file_path_for_mlflow $LOGFILE |& tee $LOGFILE
}

for recogflag in --recognition; do
  for structurePenalty in 1.5; do
    for arity in 1; do
      for task in people_gibbs_500; do
        for ppw in 10; do
          for prim in pen; do
            max_jobs 4
            sleep 1 # adding this so jobs are ordered in output
            exp $recogflag --task $task --try_all_start --partial_progress_weight $ppw --grammar $prim --arity $arity --structurePenalty $structurePenalty &
          done
        done
      done
    done
  done
done

wait
