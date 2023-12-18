# CI workflows

This directory contains workflows that are run for Continuous Integration (CI) testing.

## Static code checks

The workflow that runs static code checks is based on the clang-format and clang-tidy requirements from CMSSW. They are run automatically for commits in pull-requests, and should be roughly equivalent to running `scram build code-checks` and `scram build code-format`. They are fairly lenient, checking only the lines that were changed without making any modifications. The aim is to make sure that the code is compliant for the CMSSW integration. More information can be found [here](https://cms-sw.github.io/PRWorkflow.html). As work towards the full integration with CMSSW progresses, we will tune these checks to become more stringent and closer to what is done during the CMSSW CI checks.

## Testing worflows

The workflows that run the code and produce validation and comparison plots for the standalone and CMSSW setups depend on custom GitHub actions that are located in [SegmentLinking/TrackLooper-actions](https://github.com/SegmentLinking/TrackLooper-actions). Most of the workflow is offloaded to these actions, so it is easy to adjust the process without modifying this repository. These workflows are much more time-consuming, so they must be manually triggered by leaving a comment on the PR containing `/run standalone` and/or `/run cmssw`.

When testing the CMSSW integration, a PR in this repository might depend on a corresponding PR in the [CMSSW fork](https://github.com/SegmentLinking/cmssw). Since the two PRs need to be tested together, using `/run cmssw` would not work. For this reason, there is an optional parameter to specify a branch of the CMSSW repository, `/run cmssw some-branch`. This command needs to be in its own line in the comment or it will not work correctly.

## First-time configuration

There is a one-time configuration step that needs to be done in order to allow the CI to upload the plots to the archival repository. We leave the instructions here in case it ever needs to be done again.

1. Generate a new SSH key with `ssh-keygen -t ed25519`. Save it as `tmp_deploy_key`, and leave it without a password.
1. Create a new repository secret named `DEPLOY_PRIVATE_KEY` and set it to the contents of `tmp_deploy_key`.
1. In the repository that will store the validation plots, add a new deployment key with the contents of `tmp_deploy_key.pub`.
1. Delete both `tmp_deploy_key` and `tmp_deploy_key.pub` to prevent security issues.
