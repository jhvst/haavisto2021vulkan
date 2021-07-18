right←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/right.csv' 'UTF-8' 2 0
left←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/left.csv' 'UTF-8' 2 0
th←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/threshold.csv' 'UTF-8' 2 0
feature←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/feature.csv' 'UTF-8' 2 0
value←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/values.csv' 'UTF-8' 2 0
x←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/x.csv' 'UTF-8' 2 0
output←⎕CSV '/Users/juuso/Github/legendary-umbrella/rivi-loader/examples/dataset/output.csv' 'UTF-8' 2 0
values ← ((⍴ left)∪((2⌷⍴value)÷(1⌷⍴left))) ⍴ value
out ← (1⌷⍴output) (3⌷⍴values) ⍴ ⍬
feature ← feature+1
right ← right+1
left ← left+1
apply¨⍳(1⌷⍴out)
out ≡ output

apply←{{
    i←⍵
    node←{⍵{(⍵+1)⌷,right[⍺;],left[⍺;]}i(⍵⌷feature)⌷x≤⍵⌷th}while{⍵⌷left≠0}1
    out[i;]←node 1⌷values
}⍵}