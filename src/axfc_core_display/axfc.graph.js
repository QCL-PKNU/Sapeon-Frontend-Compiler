///////////////////////////////////////////////////////////
//   axfc.graph.js
//
//   Created: 2020. 08. 08
//
//   Authors:
//      Youngsun Han (youngsun@pknu.ac.kr)
//      Heng Sengthai (sengthai37@gmail.com)
//
//  High Performance Computing Laboratory (hpcl.pknu.ac.kr)
///////////////////////////////////////////////////////////

// config the sigma js object
//
var config = new sigma({
    container: 'container',
    type: 'svg',
    settings: {
      zoomingRatio:3,
      defaultEdgeColor: '#333',
      edgeColor: 'default',
      defaultNodeBorderColor:'#ffffff',
      borderSize:1    
    }
})

// parse the json data and initalize the graph
//
sigma.parsers.json('axfc_data.json', config ,function(s){ 
    
    // initalize side bar
    viewNodeSidebar(s.graph.nodes()[0])
    
    // check if the node is aixh_support, set the node color is red, otherwise blue
    for (const [i, node] of  s.graph.nodes().entries()) {
      s.graph.nodes()[i].color = (node.attributes.is_aixh_support ? "#c70039" : "#111d5e")
    }

    // refresh the node
    s.refresh()

    // zoom in - animation :
    sigma.misc.animation.camera(s.camera, {
      y: s.graph.nodes()[0]["read_cam0:y"] + 20,
      ratio: s.camera.ratio * s.camera.settings('zoomingRatio') * 0.03
    }, {
      duration: 1000
    })

    // set click-event on node
    s.bind('clickNode', function(e) {
      
        // set node information to Sidebar
        viewNodeSidebar(e.data.node)
    });
})

// This is method is used to set the node information to Sidebar
//
function viewNodeSidebar(node){
    document.getElementById('node_id').innerText = node.id
    document.getElementById('node_name').innerText = node.attributes.name
    document.getElementById('node_op').innerText = node.label
    document.getElementById('node_support').innerText = "aixh support: " + node.attributes.is_aixh_support
}
