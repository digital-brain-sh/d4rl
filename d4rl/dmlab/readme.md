Add new gen level to your dmlab env

Here is an example, it works for me. You may change the path if necessary. 
```bash
rm -rf ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels/old_gen_levels && \
rm -rf ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels/new_gen_levels && \
cp -r /nfs/dgx08/home/cz/sample-factory/level_gen/old_gen_levels ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels && \
cp -r /nfs/dgx08/home/cz/sample-factory/level_gen/new_gen_levels ~/miniconda3/envs/sample-factory/lib/python3.8/site-packages/deepmind_lab/baselab/game_scripts/levels
```

Following is an indicator of where to change.
```bash
rm -rf ~/${conda}/envs/${env name}/lib/${python ver}/site-packages/deepmind_lab/baselab/game_scripts/levels/old_gen_levels && \
rm -rf ~/${conda}/envs/${env name}/lib/${python ver}/site-packages/deepmind_lab/baselab/game_scripts/levels/new_gen_levels && \
cp -r /nfs/dgx08/home/cz/sample-factory/level_gen/old_gen_levels ~/${conda}/envs/${env name}/lib/${python ver}/site-packages/deepmind_lab/baselab/game_scripts/levels && \
cp -r /nfs/dgx08/home/cz/sample-factory/level_gen/new_gen_levels ~/${conda}/envs/${env name}/lib/${python ver}/site-packages/deepmind_lab/baselab/game_scripts/levels
```
