"""Runtime checks for the image-only ROI; skipped in a report-only install."""
import unittest
import numpy as np
try:
    from mmpose.datasets.transforms.image_content_bbox import ImageContentBBox
    from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine
except ImportError:
    ImageContentBBox = None


@unittest.skipIf(ImageContentBBox is None, 'Vendored model environment not installed')
class PreprocessingContract(unittest.TestCase):
    def test_annotation_bbox_cannot_change_input(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[20:100, 50:150] = 180
        outputs=[]
        for box in [[0,0,1,1],[90,70,195,119]]:
            data=ImageContentBBox()({'img':image.copy(),'bbox':np.array([box],np.float32)})
            np.testing.assert_array_equal(data['bbox'],[[45,15,154,104]])
            data=GetBBoxCenterScale(padding=1.0)(data)
            outputs.append(TopdownAffine(input_size=(768,1024),use_udp=True)(data)['img'])
        self.assertEqual(outputs[0].shape,(1024,768,3))
        np.testing.assert_array_equal(outputs[0],outputs[1])

    def test_empty_image_uses_full_image(self):
        data=ImageContentBBox()({'img':np.zeros((80,100,3),np.uint8)})
        np.testing.assert_array_equal(data['bbox'],[[0,0,99,79]])

if __name__=='__main__':unittest.main()
