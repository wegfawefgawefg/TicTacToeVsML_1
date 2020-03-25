function setup()
{
    createCanvas(windowWidth, windowHeight);

    angleMode(DEGREES)
    frameRate(1000);
    setAttributes('antialias', true);

    background(220);

    var b = nj.arange(12).reshape(4,3);   // 2d array
    console.log(b);

}

//  overflow hidden

function draw()
{
    resizeCanvas(windowWidth, windowHeight);
    background(255);

}
