# prerequisite

1. install emscripten on your environment. you can see the install procedure here: https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html
2. install node.js.

# how to try

execute following commands

```
$ cd ../../project.prj/
$ make clean && make -j8 lib_js
$ cd ../output/js
$ cp ../../project.prj/lib_js.* .
$ node benchmark.js
```
